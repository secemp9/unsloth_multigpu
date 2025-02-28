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

Run with:
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
import unsloth
from unsloth import initialize_distributed

# Initialize distributed environment with DeepSpeed
# ZeRO stages:
# - Stage 1: Optimizer state partitioning (minimal memory savings)
# - Stage 2: Optimizer + gradient partitioning (good balance between memory and speed)
# - Stage 3: Optimizer + gradient + parameter partitioning (most memory savings, but slower)
accelerator = initialize_distributed(strategy="deepspeed", zero_stage=2)

# Import after initializing distributed
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
)
from datasets import load_dataset
import torch.distributed as dist

# Get distributed rank info
rank = 0
world_size = 1
if dist.is_available() and dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

# Load model, enable 4-bit quantization for reduced VRAM
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

# Load sample dataset
dataset = load_dataset("Abirate/english_quotes", split="train")
if rank == 0:
    print(f"Full dataset size: {len(dataset)}")

# Prepare data for training
def preprocess_function(examples):
    return tokenizer(examples["quote"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments - DeepSpeed specific settings
training_args = UnslothTrainingArguments(
    output_dir="./deepspeed_results",
    per_device_train_batch_size=2,  # Smaller batch size per GPU with DeepSpeed
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    warmup_steps=10,
    max_steps=50,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    remove_unused_columns=False,
    multi_gpu_strategy="deepspeed",  # Tell Unsloth we're using DeepSpeed
    dataloader_num_workers=max(1, os.cpu_count() // world_size // 2),  # Optimal workers per GPU
    seed=42,  # Set fixed seed to ensure proper sharding
)

# Create trainer - dataset will be automatically sharded
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([f["input_ids"] for f in data]), 
                               'attention_mask': torch.stack([f["attention_mask"] for f in data])},
)

if rank == 0:
    print(f"Training on {world_size} GPUs with DeepSpeed ZeRO-2")
    
# Train the model
trainer.train()

# Save the model - only on main process
if accelerator.is_main_process:
    model.save_pretrained("./deepspeed_final_model")
    print("Model saved to ./deepspeed_final_model")
    
if rank == 0:
    print("DeepSpeed multi-GPU training complete!")