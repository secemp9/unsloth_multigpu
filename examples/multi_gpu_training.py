#!/usr/bin/env python
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example of multi-GPU training with Unsloth.

Run with:
accelerate launch --multi_gpu examples/multi_gpu_training.py

"""

import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Create training arguments first so we can pass them to initialize_distributed
from transformers import set_seed

# Set deterministic seed for reproducibility
set_seed(42)

# Import Unsloth
import unsloth

# Create training arguments
per_device_batch_size = 4
gradient_accumulation_steps = 2
learning_rate = 2e-4

# This section must come before other imports to prevent duplication issues
from unsloth import UnslothTrainingArguments

# Create training arguments object to pass to initialize_distributed
training_args = UnslothTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    lr_scheduler_type="constant",
    warmup_steps=10,
    max_steps=50,
    # Mixed precision will be set by accelerator
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    remove_unused_columns=False,
    # Tell Unsloth we're using DDP for multi-GPU
    multi_gpu_strategy="ddp",
    # Enable token counting across devices for gradient accumulation
    average_tokens_across_devices=True,
    # Other args
    seed=42,
    report_to="none",  # Disable wandb/tensorboard
    dataloader_num_workers=max(1, os.cpu_count() // 8)  # Scale workers based on available CPUs
)

# Now initialize distributed environment - pass the training args
from unsloth import initialize_distributed
accelerator = initialize_distributed(strategy="ddp", args=training_args)

# Now import the rest after initialization
from unsloth import FastLanguageModel, UnslothTrainer
from datasets import load_dataset

# Get distributed rank info
rank = 0
world_size = 1
is_main = True
if dist.is_available() and dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0

# Load model with 4-bit quantization for reduced VRAM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit", 
    max_seq_length=2048,
    dtype=None,  # Accelerate will handle mixed precision
    load_in_4bit=True,
)

# Add adapters for training
model = FastLanguageModel.get_peft_model(
    model,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    r=16,  # LoRA rank
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# Load sample dataset - each process gets the full dataset, but will shard automatically
dataset = load_dataset("Abirate/english_quotes", split="train")
if is_main:
    print(f"Full dataset size: {len(dataset)}")

# Prepare data for training
def preprocess_function(examples):
    return tokenizer(examples["quote"], truncation=True)

# Process on all ranks with optimal num_proc per rank
workers_per_rank = max(1, os.cpu_count() // world_size // 2)
tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    num_proc=workers_per_rank
)

# Use UnslothTrainer with automatic dataset sharding
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {
        'input_ids': torch.stack([f["input_ids"] for f in data]), 
        'attention_mask': torch.stack([f["attention_mask"] for f in data])
    },
)

# Sync before training to ensure all processes are ready
if world_size > 1:
    torch.distributed.barrier()

if is_main:
    print(f"Starting training on {world_size} GPUs")
    
# Train the model
trainer.train()

# Save the model on the primary process only
if accelerator.is_main_process:
    model.save_pretrained("./final_model")
    print("Model saved to ./final_model")
    
if is_main:
    print("Multi-GPU training complete!")