# Multi-GPU Training with Unsloth

This guide explains how to use Unsloth's multi-GPU training capabilities to accelerate your fine-tuning.

## Requirements

- Multiple GPUs (2+) 
- Accelerate (`pip install accelerate`)
- For DeepSpeed: `pip install deepspeed`

## Distributed Training Methods

Unsloth supports two main distributed training strategies:

1. **DDP (Distributed Data Parallel)** - Default method, simpler setup
2. **DeepSpeed** - More advanced, with ZeRO optimization levels for memory efficiency

## Basic Multi-GPU Setup (DDP)

The example script `multi_gpu_training.py` shows how to use DDP:

```python
import unsloth
from unsloth import initialize_distributed

# Initialize distributed environment
accelerator = initialize_distributed(strategy="ddp")

# Import model classes after initialization
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

# Standard model loading...

# Use the multi_gpu_strategy parameter
training_args = UnslothTrainingArguments(
    # ... other args ...
    multi_gpu_strategy="ddp",
)

# Create trainer as usual
trainer = UnslothTrainer(
    # ... standard params ...
)

# Train the model
trainer.train()

# Save only on the main process
if accelerator.is_main_process:
    model.save_pretrained("./final_model")
```

To run:
```bash
accelerate launch --multi_gpu examples/multi_gpu_training.py
```

## DeepSpeed Multi-GPU Setup

For more advanced memory optimization, the `deepspeed_multi_gpu.py` example shows how to use DeepSpeed with ZeRO:

```python
import unsloth
from unsloth import initialize_distributed

# Initialize with DeepSpeed
accelerator = initialize_distributed(strategy="deepspeed", zero_stage=2)

# ... rest of setup similar to DDP ...

# Use the deepspeed strategy
training_args = UnslothTrainingArguments(
    # ... other args ...
    multi_gpu_strategy="deepspeed",
)
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

- **Stage 1**: Optimizers state partitioning
- **Stage 2**: Adds gradient partitioning (good balance of memory and speed)
- **Stage 3**: Adds parameter partitioning (most memory efficient, but slower)

Choose the appropriate level based on your model size and GPU memory constraints.

## Tips for Multi-GPU Training

1. **Batch Size**: Adjust `per_device_train_batch_size` according to your GPU memory
2. **Gradient Accumulation**: Can be used in both DDP and DeepSpeed setups
3. **Mixed Precision**: BF16 (if supported) or FP16 is automatically used
4. **Saving**: Always check `accelerator.is_main_process` before saving to avoid conflicts

## Troubleshooting

- **NCCL Errors**: Make sure all GPUs are on the same CUDA version
- **OOM Errors**: Try reducing batch size, using gradient accumulation, or using DeepSpeed with higher ZeRO stage
- **Training Hangs**: Check for network connectivity between GPUs or NCCL timeouts

## Example Commands

Basic DDP:
```
accelerate launch --multi_gpu examples/multi_gpu_training.py
```

DeepSpeed ZeRO-2:
```
accelerate launch --config_file ds_config.yaml examples/deepspeed_multi_gpu.py
```

Select specific GPUs (example: only use GPU 0 and 2):
```
CUDA_VISIBLE_DEVICES=0,2 accelerate launch --multi_gpu examples/multi_gpu_training.py
```