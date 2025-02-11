
# %%
import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, set_seed
set_seed(42)
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from trl import PPOTrainer
from trl.core import LengthSampler
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from utils import build_dataset, build_model, build_config, train, build_test_case, test
# %%
import argparse
parser = argparse.ArgumentParser(description='Train a model with specified learning rate and clipping norm.')
parser.add_argument('--reward-model', type=str, default='mingxilei/dpsgd_filter_imdb_reward_8.0_0.001')
parser.add_argument('--sft-model', type=str, default='/root/autodl-tmp/llama-SFT')
parser.add_argument('--gt-reward-model', type=str, default="siebert/sentiment-roberta-large-english")
parser.add_argument('--project', type=str, default='imdb-llama')
parser.add_argument('--kl', type=float, default=0.1)
args = parser.parse_args()


config, sent_kwargs = build_config(args.sft_model, args.kl)
dataset = build_dataset(config)
model, ref_model, tokenizer = build_model(config.model_name)
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
test_case = build_test_case(config, collator, 100)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.RMSprop(params, lr=config.learning_rate)

# %%
import wandb
wandb.init(project=args.project, name=args.reward_model)
wandb.run.log_code(".")
ppo_trainer = PPOTrainer(
    config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer, #lr_scheduler=scheduler
)
test_case = ppo_trainer.accelerator.prepare(test_case)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
sentiment_pipe = pipeline("sentiment-analysis", model=args.reward_model, device=device)
gt_sentiment_pipe = pipeline("sentiment-analysis", model=args.gt_reward_model, device=device)

# %%
text = "this movie was really bad!!"
sentiment_pipe(text, **sent_kwargs)
text = "this movie was really good!!"
gt_sentiment_pipe(text, **sent_kwargs)

# %% [markdown]
# The training loop consists of the following main steps:
# 1. Get the query responses from the policy network (GPT-2)
# 2. Get sentiments for query/responses from BERT
# 3. Optimize policy with PPO using the (query, response, reward) triplet

# %%
output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)


generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

test_log_every = 10
for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
    stats, batch, gt_rewards = train(ppo_trainer, batch, output_length_sampler, generation_kwargs, tokenizer, sentiment_pipe, sent_kwargs, gt_sentiment_pipe)
    if (epoch + 1) % test_log_every == 0:
        with torch.no_grad():
            test_batch = next(iter(test_case))
            test_rewards, kl = test(ppo_trainer, test_batch, output_length_sampler, generation_kwargs, tokenizer, sent_kwargs, gt_sentiment_pipe)
        stats["eval/rewards_mean"] = torch.tensor(test_rewards).mean().numpy().item()
        stats["eval/rewards_std"] = torch.tensor(test_rewards).std().numpy().item()
        stats["eval/kl"] = kl
    ppo_trainer.log_stats(stats, batch, gt_rewards)
    # break

# %%
wandb.finish()
