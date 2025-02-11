
import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

def build_dataset(
    config,
    dataset_name="stanfordnlp/imdb",
    input_min_text_length=2,
    input_max_text_length=8,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def build_model(model_name):
    sft_model = AutoPeftModelForCausalLM.from_pretrained(model_name).merge_and_unload()
    lora_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM", lora_dropout=0.1)
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model, peft_config=lora_config)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model, peft_config=lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    
    return model, ref_model, tokenizer

def build_config(sft_model, kl):
    config = PPOConfig(
        model_name=sft_model,
        learning_rate=1e-5,
        log_with="wandb",
        adap_kl_ctrl=False,
        # target=0.5,
        batch_size=64,
        mini_batch_size=64,
        init_kl_coef=kl,
    )

    sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
    return config, sent_kwargs

def build_test_case(
    config,
    data_collator,
    num_sample,
    dataset_name="stanfordnlp/imdb",
    input_min_text_length=2,
    input_max_text_length=8,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="test")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)
    ds = ds.select(range(num_sample))
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=num_sample,
            collate_fn=data_collator,
            shuffle=False,
            drop_last=True,
        )
    return dataloader

def train(ppo_trainer, batch, output_length_sampler, generation_kwargs, tokenizer, sentiment_pipe, sent_kwargs, gt_sentiment_pipe):
    query_tensors = batch["input_ids"]

    #### Get response
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        query_response = ppo_trainer.generate(query, **generation_kwargs).squeeze()
        response_len = len(query_response) - len(query)
        response_tensors.append(query_response[-response_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    positive_scores = [
        item["score"]
        for output in pipe_outputs
        for item in output
        if item["label"] == "POSITIVE"
    ]
    rewards = [torch.tensor(score) for score in positive_scores]

    pipe_outputs = gt_sentiment_pipe(texts, **sent_kwargs)
    positive_scores = [
        item["score"]
        for output in pipe_outputs
        for item in output
        if item["label"] == "POSITIVE"
    ]
    gt_rewards = [torch.tensor(score) for score in positive_scores]

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    return stats, batch, gt_rewards

def test(ppo_trainer, batch, output_length_sampler, generation_kwargs, tokenizer, sent_kwargs, gt_sentiment_pipe):
    query_tensors = batch["input_ids"]

    #### Get response
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        query_response = ppo_trainer.generate(query, **generation_kwargs).squeeze()
        outputs = ppo_trainer.model.generate(query.unsqueeze(0), **generation_kwargs, return_dict_in_generate=True, output_scores=True)
        transition_scores = ppo_trainer.model.pretrained_model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        response_len = len(query_response) - len(query)
        response_tensors.append(query_response[-response_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]

    pipe_outputs = gt_sentiment_pipe(texts, **sent_kwargs)
    positive_scores = [
        item["score"]
        for output in pipe_outputs
        for item in output
        if item["label"] == "POSITIVE"
    ]
    gt_rewards = [torch.tensor(score) for score in positive_scores]
    kl = get_kl(ppo_trainer, query_tensors, response_tensors, gt_rewards)
    return gt_rewards, kl

def get_kl(ppo_trainer, queries, responses, rewards):
    model_inputs = ppo_trainer.prepare_model_inputs(queries, responses)
    full_kl_penalty = True
    _, logits_or_none, _, masks = ppo_trainer.batched_forward_pass(
        ppo_trainer.model,
        queries,
        responses,
        model_inputs,
        response_masks=None,
        return_logits=full_kl_penalty,
    )
    with ppo_trainer.optional_peft_ctx():
        _, ref_logits_or_none, _, _ = ppo_trainer.batched_forward_pass(
                        ppo_trainer.ref_model,
                        queries,
                        responses,
                        model_inputs,
                        return_logits=full_kl_penalty,
                    )

    active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
    ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

    
    kls = []
    for logprob, ref_logprob in zip(active_full_logprobs, ref_full_logprobs):
        # compute KL penalty (from difference in logprobs)
        kl = F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)
        kls.append(kl)
    kls = torch.stack(kls)
    kl_list = ((kls) * masks).sum(axis=-1)
    mean_kl = kl_list.mean()
    return mean_kl


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy
