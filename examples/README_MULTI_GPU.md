# Multi-GPU Training with Unsloth

This guide explains how to use Unsloth's multi-GPU training capabilities to accelerate your fine-tuning. The implementation ensures proper data sharding and gradient synchronization across GPUs to prevent duplication issues.

## Requirements

- Multiple GPUs (2+) 
- Accelerate (`pip install accelerate`)
- For DeepSpeed: `pip install deepspeed`

## Distributed Training Methods

Unsloth supports two main distributed training strategies:

1. **DDP (Distributed Data Parallel)** - Default method, simpler setup
2. **DeepSpeed** - More advanced, with ZeRO optimization levels for memory efficiency

## Important: Import Order

When using multi-GPU training, **import order matters**. For best results, follow this pattern:

```python
# First imports
import torch
import torch.distributed as dist
from transformers import set_seed
set_seed(42)  # Set seed for reproducibility

# Import Unsloth
import unsloth

# Create training arguments first
from unsloth import UnslothTrainingArguments
training_args = UnslothTrainingArguments(
    # your arguments...
    multi_gpu_strategy="ddp",
)

# Now initialize distributed environment
from unsloth import initialize_distributed
accelerator = initialize_distributed(strategy="ddp", args=training_args)

# Import the rest after initialization
from unsloth import FastLanguageModel, UnslothTrainer
# Continue with your code...
```

This order prevents duplication issues when running on multiple GPUs.

## Basic Multi-GPU Setup (DDP)

The example script `multi_gpu_training.py` shows how to use DDP properly:

```python
import torch
import unsloth
from unsloth import UnslothTrainingArguments

# Create training args first
training_args = UnslothTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    # Other args...
    multi_gpu_strategy="ddp",  # Tell Unsloth we're using DDP
)

# Initialize distributed environment - pass the training args
from unsloth import initialize_distributed
accelerator = initialize_distributed(strategy="ddp", args=training_args)

# Import the rest after initialization
from unsloth import FastLanguageModel, UnslothTrainer
from datasets import load_dataset

# Rest of your code...
```

To run:
```bash
accelerate launch --multi_gpu examples/multi_gpu_training.py
```

## DeepSpeed Multi-GPU Setup

For more advanced memory optimization and better scaling across many GPUs, use DeepSpeed with ZeRO:

```python
import unsloth
from unsloth import UnslothTrainingArguments

# Set ZeRO stage
zero_stage = 2  # Choose 0, 1, 2, or 3 based on your needs

# Create training args first
training_args = UnslothTrainingArguments(
    output_dir="./deepspeed_results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    # Other args...
    multi_gpu_strategy="deepspeed",  # Tell Unsloth we're using DeepSpeed
)

# Initialize with DeepSpeed
from unsloth import initialize_distributed
accelerator = initialize_distributed(
    strategy="deepspeed", 
    zero_stage=zero_stage,
    args=training_args
)

# Now import the rest and continue...
```

To run with DeepSpeed, create a config file first:

```yaml
# ds_config.yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 2
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero_stage: 2
  train_micro_batch_size_per_gpu: 2
distributed_type: DEEPSPEED
```

Then run:
```bash
accelerate launch --config_file ds_config.yaml examples/deepspeed_multi_gpu.py
```

## ZeRO Stages (DeepSpeed)

DeepSpeed offers 3 stages of ZeRO optimization:

- **Stage 1**: Optimizers state partitioning (minimal memory savings)
- **Stage 2**: Adds gradient partitioning (good balance of memory and speed)
- **Stage 3**: Adds parameter partitioning (most memory efficient, but slower)

Choose the appropriate level based on your model size and GPU memory constraints.

## Dataset Sharding

Unsloth automatically shards your dataset across GPUs to prevent duplication. Our implementation uses `datasets.distributed.split_dataset_by_node` to ensure each GPU processes a unique portion of the data.

You don't need to manually shard the dataset - just pass it to the trainer as usual:

```python
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,  # Will be automatically sharded
    # Other params...
)
```

## Proper Gradient Accumulation

Unsloth handles gradient accumulation correctly in distributed settings by synchronizing losses and gradients at the right points. This ensures that:

1. Each GPU processes a different part of the dataset
2. Gradients are properly synchronized
3. The effective batch size is: `per_device_batch_size × num_gpus × gradient_accumulation_steps`

## Tips for Multi-GPU Training

1. **Batch Size**: Adjust `per_device_train_batch_size` according to your GPU memory
2. **Gradient Accumulation**: Increase `gradient_accumulation_steps` to simulate larger batch sizes
3. **Worker Scaling**: Set `dataloader_num_workers` scaled by the number of GPUs
4. **Deterministic Training**: Set a fixed seed (e.g., `seed=42`) for reproducible results
5. **Saving**: Always check `accelerator.is_main_process` before saving to avoid conflicts

## Troubleshooting

- **Duplicate Messages**: If you see duplicate initialization messages, check your import order
- **NCCL Errors**: Make sure all GPUs are on the same CUDA version
- **OOM Errors**: Try reducing batch size, using gradient accumulation, or using DeepSpeed with higher ZeRO stage
- **Training Hangs**: Check for network connectivity between GPUs or NCCL timeouts
- **Different GPU Sizes**: If using GPUs with different memory sizes, adjust `per_device_train_batch_size` to fit the smallest GPU

## Example Commands

Basic DDP:
```
accelerate launch --multi_gpu examples/multi_gpu_training.py
```

DeepSpeed ZeRO-2:
```
accelerate launch --config_file ds_config.yaml examples/deepspeed_multi_gpu.py
```

Select specific GPUs:
```
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu examples/multi_gpu_training.py
```

Force BF16 precision (if supported):
```
accelerate launch --multi_gpu --mixed_precision=bf16 examples/multi_gpu_training.py
```