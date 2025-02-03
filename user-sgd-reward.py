# %%
import os
from datasets import load_dataset
import evaluate
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
import torch
from peft import get_peft_model
from peft import LoraConfig
import numpy as np
import random
from torch import nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.optim as optim
from opacus.accountants.utils import get_noise_multiplier

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
parser.add_argument('--eps', type=float, default=8, help='epsilon')
args = parser.parse_args()

# %%
def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
    
def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    # No gradient calculation for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits.squeeze(-1)
            loss = BCEWithLogitsLoss()(logits, labels.float())
            total_loss += loss.item()

            # Calculate predictions
            predictions = torch.sigmoid(torch.tensor(logits)) > 0.5
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy
    
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 512
    
# Initialize classifier parameters with Kaiming initialization
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

d = count_trainable_parameters(model)
print(d)

ds = load_dataset("imdb")
model = model.cuda()
def tokenize(examples):
    outputs = tokenizer(examples['text'], truncation=True)
    return outputs
def convert_labels_to_float(example):
    example["label"] = example["label"].float()
    return example
tokenized_ds = ds.map(tokenize, batched=True).shuffle(seed=42)
tokenized_ds.set_format("torch")
tokenized_ds = tokenized_ds.remove_columns(["text"])

# %%
from torch.utils.data import Sampler, BatchSampler
import itertools

num_users = 2500
m = len(tokenized_ds["train"]) // num_users
user_to_index = list()
for i in range(num_users):
    user_to_index.append(list(range(i*m, (i+1)*m)))


class UserSampler(Sampler):
    def __init__(self, user_to_index):
        self.user_to_index = user_to_index
        random.shuffle(self.user_to_index)
        self.total_index = list(itertools.chain(*self.user_to_index))
    def __iter__(self):
        return iter(self.total_index)
    def __len__(self):
        return len(self.total_index)

data_collator = DataCollatorWithPadding(tokenizer)
user_batch_size = 50
train_dataloader = DataLoader(
    tokenized_ds["train"], batch_size=user_batch_size*m,
    collate_fn=data_collator, sampler=UserSampler(user_to_index)
)
test_dataloader = DataLoader(
    tokenized_ds["test"], shuffle=False, batch_size=64, collate_fn=data_collator
)
# %%
lr = args.lr
optimizer = SGD(model.parameters(), lr=lr)
# Define linear warmup scheduler
epochs = 10
steps = len(train_dataloader) * epochs
warmup_steps = steps // 5
warmup_scheduler = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_steps)
# Define cosine annealing scheduler
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=steps-warmup_steps)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

# %%
target_epsilon = args.eps
target_delta = 1e-5
sample_rate = user_batch_size / num_users
noise_multiplier = get_noise_multiplier(target_epsilon=target_epsilon, target_delta=target_delta, sample_rate=sample_rate, epochs=epochs)
print(f"noise_multiplier: {noise_multiplier}")

# %%
from torch.nn.utils import clip_grad_norm_
total_loss = 0
criterion = BCEWithLogitsLoss()
max_grad_norm = 1
lr_record = []

for _ in range(epochs):
    #train
    model.train()
    for batch in tqdm(train_dataloader):
        for param in model.parameters():
            if param.requires_grad:
                param.accumulated_grads = []

        batch = {k: torch.split(v.to('cuda'), m) for k, v in batch.items()}
        for i in range(user_batch_size):
            uesr_batch = {k: v[i] for k, v in batch.items()}
    
            # Forward pass
            labels = uesr_batch.pop("labels")
            outputs = model(**uesr_batch)
            logits = outputs.logits.squeeze(-1)
            loss = BCEWithLogitsLoss()(logits, labels.float())

            # User averaged gradient
            optimizer.zero_grad()
            loss.backward()
            # Clip per-user gradients
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            for param in model.parameters():
                if param.requires_grad:
                    per_sample_grad = param.grad.clone().detach()
                    param.accumulated_grads.append(per_sample_grad)

        
        # Aggregate back
        for param in model.parameters():
            if param.requires_grad:
                param.accumulated_grads = torch.stack(param.accumulated_grads, dim=0)
                noise = torch.normal(mean=0.0, std=noise_multiplier * max_grad_norm,
                                    size=param.grad.size(),
                                    device=param.grad.device)
                param.grad = (param.accumulated_grads.sum(dim=0) + noise) / user_batch_size

        # Now we are ready to update and add noise!
        for param in model.parameters():
            lr = optimizer.param_groups[0]['lr']
            if param.requires_grad:
                param.data = param.data.add_(param.grad, alpha=-lr)

        scheduler.step()

    avg_loss, accuracy = evaluate_model(model, test_dataloader, 'cuda')
        
model.push_to_hub("dpsgd_imdb_reward_{}_{}".format(target_epsilon, args.lr))
