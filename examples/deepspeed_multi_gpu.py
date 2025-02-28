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
Example of DeepSpeed multi-GPU training with Unsloth.

This example demonstrates using DeepSpeed ZeRO for distributed training.

Run this example with:
accelerate launch --config_file ds_config.yaml examples/deepspeed_multi_gpu.py

Create a ds_config.yaml file with:
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 2
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero_stage: 2
  train_micro_batch_size_per_gpu: 2
distributed_type: DEEPSPEED
"""

import os
import torch
import torch.distributed as dist
from transformers import set_seed

# Set seed for reproducibility
set_seed(42)

# Import Unsloth
import unsloth

# Define training parameters
per_device_batch_size = 2  # Smaller batch size per GPU with DeepSpeed
gradient_accumulation_steps = 2
learning_rate = 2e-4
zero_stage = 2  # ZeRO stage to use (0, 1, 2, or 3)

# Create training arguments first to pass to initialize_distributed
from unsloth import UnslothTrainingArguments

# Create training arguments object
training_args = UnslothTrainingArguments(
    output_dir="./deepspeed_results",
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    lr_scheduler_type="constant",
    warmup_steps=10,
    max_steps=50,
    # Mixed precision will be set by DeepSpeed
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    remove_unused_columns=False,
    # Tell Unsloth we're using DeepSpeed
    multi_gpu_strategy="deepspeed",
    # Ensure proper token counting with gradient accumulation
    average_tokens_across_devices=True,
    # Other args
    seed=42,
    report_to="none",  # Disable wandb/tensorboard
    dataloader_num_workers=max(1, os.cpu_count() // 8)  # Scale workers based on available CPUs
)

# Initialize distributed environment with DeepSpeed - pass the training args
from unsloth import initialize_distributed
accelerator = initialize_distributed(
    strategy="deepspeed", 
    zero_stage=zero_stage,
    args=training_args
)

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

# Load model, enable 4-bit quantization for reduced VRAM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit", 
    max_seq_length=2048,
    dtype=None,  # DeepSpeed will handle mixed precision
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

# Create trainer with DeepSpeed configuration
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
    print(f"Training on {world_size} GPUs with DeepSpeed ZeRO-{zero_stage}")
    print(f"Global batch size: {per_device_batch_size * world_size * gradient_accumulation_steps}")
    
# Train the model
trainer.train()

# Save the model - only on main process
if accelerator.is_main_process:
    model.save_pretrained("./deepspeed_final_model")
    print("Model saved to ./deepspeed_final_model")
    
if is_main:
    print("DeepSpeed multi-GPU training complete!")