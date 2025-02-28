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
import unsloth
from unsloth import initialize_distributed

# Initialize distributed environment
accelerator = initialize_distributed(strategy="ddp")

# This must be imported after initializing the distributed environment
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
)
from datasets import load_dataset

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

# Prepare data for training
def preprocess_function(examples):
    return tokenizer(examples["quote"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = UnslothTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
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
    multi_gpu_strategy="ddp",  # Tell Unsloth we're using DDP
)

# Create trainer
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([f["input_ids"] for f in data]), 
                               'attention_mask': torch.stack([f["attention_mask"] for f in data])},
)

print(f"Training on {accelerator.num_processes} GPUs")
# Train the model
trainer.train()

# Save the model on the primary process only
if accelerator.is_main_process:
    model.save_pretrained("./final_model")
    
print("Multi-GPU training complete!")