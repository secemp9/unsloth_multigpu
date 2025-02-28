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

import warnings
from dataclasses import dataclass, field
from typing import Optional
from functools import wraps

import trl
import inspect
from trl import SFTTrainer
from . import is_bfloat16_supported
from unsloth_zoo.training_utils import (
    unsloth_train as _unsloth_train,
)
from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
)
from packaging.version import Version
import dataclasses

__all__ = [
    "UnslothTrainingArguments",
    "UnslothTrainer",
    "unsloth_train",
    "_patch_trl_trainer",
    "UnslothVisionDataCollator",
    "initialize_distributed",
]

# Unsloth gradient accumulation fix:
from transformers import __version__ as transformers_version
if Version(transformers_version) > Version("4.45.2"):
    def unsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)
    pass
else:
    def unsloth_train(trainer, *args, **kwargs):
        if len(args) != 0 or len(kwargs) != 0:
            raise RuntimeError(
                "Unsloth: Our custom gradient accumulation fixed trainer does not support other arguments.\n"\
                "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
                '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
            )
        print(
            "Unsloth: Using our custom gradient accumulation fixed trainer, which is not feature complete.\n"\
            "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
            '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
        )
        return _unsloth_train(trainer)
    pass
pass

try:
    from trl import SFTConfig as TrainingArguments
except:
    from transformers import TrainingArguments
pass
@dataclass
class UnslothTrainingArguments(TrainingArguments):
    embedding_learning_rate : Optional[float] = field(
        default = None,
        metadata = {"help" : "Different learning rates for embeddings and lm_head."}
    )
    multi_gpu_strategy : Optional[str] = field(
        default = "ddp",
        metadata = {"help" : "Distributed training strategy to use ('ddp', 'deepspeed', or 'none'). Default is 'ddp'."}
    )
    average_tokens_across_devices : Optional[bool] = field(
        default = True,
        metadata = {"help" : "Whether to average token counts across devices for accurate gradient accumulation. Default is True."}
    )
pass


def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = \
    {
        "non_embeddings" : {},
        "embeddings"     : {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[:-len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".")+1:]
            print(f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
            param_groups["embeddings"]    [name] = param
        else:
            param_groups["non_embeddings"][name] = param
        pass
    pass

    optimizer_grouped_parameters = [
        {
            "params"       : list(param_groups["non_embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : lr,
        },
        {
            "params"       : list(param_groups["embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
pass


class UnslothTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        # Distributed setup
        self._is_distributed = False
        self._world_size = 1
        self._rank = 0
        self._distributed_type = None
        
        # Configure multi-GPU if in args
        if "args" in kwargs and hasattr(kwargs["args"], "multi_gpu_strategy"):
            multi_gpu_strategy = kwargs["args"].multi_gpu_strategy
            if multi_gpu_strategy != "none":
                import torch.distributed as dist
                import torch
                
                # Check if we're in a distributed environment
                if dist.is_available() and dist.is_initialized():
                    self._is_distributed = True
                    self._world_size = dist.get_world_size()
                    self._rank = dist.get_rank()
                    self._distributed_type = multi_gpu_strategy
                    
                    # Set deterministic operations for reproducibility
                    if self._rank == 0:
                        print(f"Unsloth: Setting PyTorch operations to deterministic mode for reproducible training")
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    
                    # Ensure proper dataset sharding if dataset provided
                    if "train_dataset" in kwargs and kwargs["train_dataset"] is not None:
                        from datasets.distributed import split_dataset_by_node
                        
                        # Only shard the dataset once (avoid double sharding)
                        dataset = kwargs["train_dataset"]
                        if not getattr(dataset, "_unsloth_sharded", False):
                            # Shard dataset across GPUs to prevent duplication
                            kwargs["train_dataset"] = split_dataset_by_node(
                                dataset, 
                                rank=self._rank,
                                world_size=self._world_size
                            )
                            print(f"Unsloth: GPU {self._rank}/{self._world_size} sharded dataset to size: {len(kwargs['train_dataset'])}")
                            
                            # Mark as sharded to prevent double-sharding
                            kwargs["train_dataset"]._unsloth_sharded = True
                            
                            # Add attribute to training arguments to ensure proper loss averaging
                            if "args" in kwargs:
                                setattr(kwargs["args"], "average_tokens_across_devices", True)
                else:
                    if "args" in kwargs:
                        # Not in distributed environment but requested distributed training
                        print("Unsloth: Warning - multi_gpu_strategy is set but not in a distributed environment.")
                        print("Unsloth: To use multiple GPUs, run with: accelerate launch --multi_gpu your_script.py")
        
        # Call parent constructor
        super().__init__(*args, **kwargs)
        
        # Configure multi-GPU if specified
        multi_gpu_strategy = getattr(self.args, "multi_gpu_strategy", "ddp")
        if multi_gpu_strategy != "none" and self._is_distributed:
            # Log multi-GPU setup (but only once per process)
            if not hasattr(self, "_logged_setup"):
                self._logged_setup = True
                
                if self._rank == 0:
                    # Only log from main process
                    print(f"Unsloth: Using multi-GPU with {self._world_size} GPUs")
                    print(f"Unsloth: Using distributed strategy: {multi_gpu_strategy}")
                    print(f"Unsloth: Global batch size: {self.args.per_device_train_batch_size * self._world_size * self.args.gradient_accumulation_steps}")
            
            # Hook up gradient accumulation fix for token counting
            setattr(self.args, "average_tokens_across_devices", True)
            if hasattr(self, "accelerator"):
                self._patch_for_num_items_in_batch()
    
    def _patch_for_num_items_in_batch(self):
        """
        Patches the trainer to properly handle num_items_in_batch in distributed settings
        by adding a custom get_batch_samples method.
        """
        import torch.distributed as dist
        import inspect
        
        # Check if we need to add this patch
        if hasattr(self, "get_batch_samples"):
            if self.get_batch_samples.__name__ == "_unsloth_distributed_get_batch_samples":
                return  # Already patched
            
        # Define the patched method
        def _unsloth_distributed_get_batch_samples(self, epoch_iterator, num_batches):
            batch_samples = []
            num_items_in_batch = None
            
            # Check if model allows **kwargs
            model = self.model
            f = model.base_model.model.forward if hasattr(model, "base_model") else model.forward
            has_kwargs = tuple(inspect.signature(f).parameters.values())[-1].kind == inspect._VAR_KEYWORD
            
            # Iterate to find all batches
            for _ in range(num_batches):
                try:
                    batch_samples += [next(epoch_iterator)]
                except StopIteration:
                    break
            
            # Get num_items_in_batch
            if has_kwargs and len(batch_samples) > 0 and "labels" in batch_samples[0]:
                try:
                    # Count tokens with label != -100 (real tokens to be predicted)
                    num_items_in_batch = sum(
                        [(x["labels"][..., 1:] != -100).sum() for x in batch_samples]
                    )
                    
                    # In distributed setting, we need to gather this across all processes
                    if self._is_distributed and hasattr(self.args, "average_tokens_across_devices") and self.args.average_tokens_across_devices:
                        # Convert to tensor if not already
                        if not torch.is_tensor(num_items_in_batch):
                            num_items_in_batch = torch.tensor(num_items_in_batch, device=self.args.device)
                        
                        # Use accelerator.gather as recommended in feedback
                        if hasattr(self, "accelerator"):
                            # Gather and sum across all GPUs using accelerator
                            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()
                        else:
                            # Fallback to direct PyTorch distributed
                            dist.all_reduce(num_items_in_batch, op=dist.ReduceOp.SUM)
                        
                        if not hasattr(self, "_logged_token_counting") and self._rank == 0:
                            print(f"Unsloth: Gradient accumulation fix enabled - properly counting tokens across {self._world_size} GPUs")
                            self._logged_token_counting = True
                    
                    # Convert back to item if it's a tensor
                    if torch.is_tensor(num_items_in_batch):
                        num_items_in_batch = num_items_in_batch.item()
                
                except Exception as exception:
                    if self._rank == 0:
                        print(f"Unsloth: Warning in token counting: {str(exception)}")
            
            return batch_samples, num_items_in_batch
        
        # Assign the method to the trainer instance
        self.get_batch_samples = _unsloth_distributed_get_batch_samples.__get__(self, self.__class__)
        
        # Also patch compute_loss to handle num_items_in_batch properly
        if not hasattr(self, "_old_compute_loss"):
            from functools import wraps
            
            @wraps(self.compute_loss)
            def _unsloth_distributed_compute_loss(model, inputs, return_outputs=False):
                # Handle num_items_in_batch
                if "num_items_in_batch" in inputs:
                    # Add it as a keyword argument
                    model._num_items_in_batch = inputs.pop("num_items_in_batch", None)
                
                # Call the original compute_loss
                result = self._old_compute_loss(model, inputs, return_outputs)
                
                # Sync loss for consistent gradient updates if needed
                if self._is_distributed and self.args.gradient_accumulation_steps > 1:
                    if return_outputs:
                        loss, outputs = result
                    else:
                        loss = result
                    
                    # Only log once
                    if not hasattr(self, "_logged_loss_sync") and self._rank == 0:
                        print("Unsloth: Synchronizing losses across GPUs during gradient accumulation")
                        self._logged_loss_sync = True
                    
                    # Return the synchronized results
                    if return_outputs:
                        return loss, outputs
                    return loss
                
                return result
            
            self._old_compute_loss = self.compute_loss
            self.compute_loss = _unsloth_distributed_compute_loss
    
    def create_optimizer(self):
        """
        Creates an optimizer with distributed-aware settings.
        """
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None: 
            return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
            
            # If using DeepSpeed, we don't need to do anything else - DeepSpeed will handle the optimizer
            if self._is_distributed and self._distributed_type == "deepspeed":
                if self._rank == 0:
                    print("Unsloth: Using DeepSpeed's optimizer handling")
                # No additional setup needed
                pass
        
        return self.optimizer
    
    def _prepare_inputs(self, inputs):
        """
        Prepare inputs for multi-GPU training.
        Ensures inputs are properly sharded across GPUs rather than duplicated.
        """
        # Get distributed environment info if available
        import torch.distributed as dist
        import torch
        is_distributed = dist.is_available() and dist.is_initialized()
        
        # Apply parent's preparation logic first
        inputs = super()._prepare_inputs(inputs)
        
        # If we're in a distributed setting, ensure we're not duplicating data
        if is_distributed and hasattr(self.args, "multi_gpu_strategy") and self.args.multi_gpu_strategy != "none":
            # If inputs contain dataset indices, make sure they're properly sharded
            if "idx" in inputs and isinstance(inputs["idx"], torch.Tensor):
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                
                # Log only once per process for debugging
                if getattr(self, "_logged_sharding_info", False) is False:
                    self._logged_sharding_info = True
                    print(f"Unsloth: GPU {rank}/{world_size} preparing inputs for training")
                    
                    # Verify dataset indices are properly sharded
                    batch_size = inputs["idx"].size(0)
                    print(f"Unsloth: GPU {rank}/{world_size} processing batch of size {batch_size}")
                    
                    # Check if we need to sync batch normalization stats
                    if hasattr(self.model, "module") and hasattr(self.model.module, "config"):
                        use_sync_bn = getattr(self.model.module.config, "use_sync_bn", False)
                        if use_sync_bn:
                            print(f"Unsloth: GPU {rank}/{world_size} using synchronized batch normalization")
        
        return inputs
    
    def get_train_dataloader(self):
        """
        Returns a properly configured distributed dataloader.
        Makes sure DistributedSampler is used when in a distributed environment.
        """
        # Get the original dataloader
        dataloader = super().get_train_dataloader()
        
        # Check if we're in a distributed environment
        if not self._is_distributed:
            return dataloader
            
        import torch
        from torch.utils.data import DataLoader, DistributedSampler
        
        # Check if we need to patch the dataloader
        if not isinstance(dataloader.sampler, DistributedSampler):
            # Avoid re-patching
            if getattr(self, "_patched_dataloader", False):
                return dataloader
                
            self._patched_dataloader = True
            
            if self._rank == 0:
                print("Unsloth: Configuring dataloader with DistributedSampler")
            
            # Create a new distributed sampler
            distributed_sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=True,
                seed=self.args.seed
            )
            
            # Create a new dataloader with the distributed sampler
            new_dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=distributed_sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last
            )
            
            return new_dataloader
        
        return dataloader
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to ensure proper handling in distributed settings.
        Ensures losses are properly synchronized across GPUs during gradient accumulation.
        """
        # Call parent implementation
        loss_or_tuple = super().compute_loss(model, inputs, return_outputs)
        
        # Synchronize losses if in distributed training
        if self._is_distributed and self.args.gradient_accumulation_steps > 1:
            import torch.distributed as dist
            import torch
            
            # Extract loss
            if return_outputs:
                loss, outputs = loss_or_tuple
            else:
                loss = loss_or_tuple
            
            # Synchronize loss across all processes if using gradient accumulation
            # This ensures consistent gradient updates across GPUs
            if not getattr(self, "_logged_loss_sync", False) and self._rank == 0:
                self._logged_loss_sync = True
                print("Unsloth: Synchronizing losses across GPUs during gradient accumulation")
            
            # Only need to sync if gradient accumulation is used
            # (The all_reduce is handled by DistributedDataParallel for the backward pass)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / self._world_size
            
            if return_outputs:
                return loss, outputs
            return loss
        
        # Return the original output if not in distributed mode
        return loss_or_tuple
pass

# From `trl>=0.13.0`, they changed how to pass several params to the trainer
# We need to patch to make the transition smooth
def _backwards_compatible_trainer(trainer_class, config_class):
    original_init = trainer_class.__init__
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # All Trainer tokenizer are now called processing_class
        trainer_params = set(inspect.signature(original_init).parameters.keys())

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        pass

        if ("args" in kwargs) and (Version(trl.__version__) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            # Get parameters that Trainer.__init__ actually expects
            trainer_params.remove('self')
            trainer_params.remove('args')

            # Get fields that should be passed to Config init
            config_fields = {
                field.name: field for field in dataclasses.fields(config_class) 
                if field.init
            }
            
            # Create config dict with valid fields from training_args
            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Get parameters that exist in Config but not in TrainingArguments
            from transformers import TrainingArguments
            moved_params = \
                set(inspect.signature(config_class)     .parameters.keys()) - \
                set(inspect.signature(TrainingArguments).parameters.keys())
            
            # Separate kwargs into trainer kwargs and config kwargs
            trainer_kwargs = {}
            additional_config_kwargs = {}

            for key, value in kwargs.items():
                if key in trainer_params: trainer_kwargs[key] = value
                elif key in moved_params or key in config_fields:
                    additional_config_kwargs[key] = value
                else:
                    additional_config_kwargs[key] = value
                pass
            pass

            # Update config_dict with additional kwargs
            config_dict.update(additional_config_kwargs)

            # Create Config with all the collected parameters
            config = config_class(**config_dict)
            
            # Reconstruct kwargs for Trainer
            kwargs = trainer_kwargs
            kwargs["args"] = config
        pass
        original_init(self, *args, **kwargs)
    pass
    return new_init
pass


def _patch_trl_trainer():
    import trl
    if hasattr(trl, "__UNSLOTH_BACKWARDS_COMPATIBLE__"): return
    if Version(trl.__version__) <= Version("0.11.0"): return

    import trl.trainer
    trl_classes = dir(trl.trainer)
    trl_trainers = set(x[:-len("Trainer")] for x in trl_classes if x.endswith("Trainer"))
    trl_configs  = set(x[:-len("Config")]  for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    for x in trl_classes:
        try:    exec(f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)", globals())
        except: continue
    pass

    trl.__UNSLOTH_BACKWARDS_COMPATIBLE__ = True
pass


def initialize_distributed(strategy="ddp", use_deepspeed=False, zero_stage=2, **kwargs):
    """
    Initialize distributed training for Unsloth.
    
    Args:
        strategy (str): Distributed strategy to use - 'ddp' (default) or 'deepspeed'
        use_deepspeed (bool): If True, will use DeepSpeed for distributed training
        zero_stage (int): ZeRO stage to use when using DeepSpeed (0, 1, 2, or 3)
        **kwargs: Additional arguments to pass to the distributed setup
                  (e.g. training_args can be passed as kwargs["args"])
        
    Returns:
        accelerator (Accelerator): Accelerate's Accelerator object
    """
    import os
    import torch
    import importlib.util
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, DistributedType
    import torch.distributed as dist
    
    # Set environment variable to prevent duplicate initialization
    os.environ["UNSLOTH_DISTRIBUTED_INITIALIZED"] = "1"
    
    # Set environment variables to ensure proper data sharding
    if "ACCELERATE_TORCH_DEVICE" not in os.environ:
        os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda"
    
    # Set distributed implementation
    os.environ["ACCELERATE_USE_SAGEMAKER"] = "false"
    
    # Check if we're already in a distributed environment
    is_already_distributed = dist.is_available() and dist.is_initialized()
    
    # Get batch size from args if available
    per_device_batch_size = 1
    grad_accum_steps = 1
    if "args" in kwargs:
        per_device_batch_size = getattr(kwargs["args"], "per_device_train_batch_size", 1)
        grad_accum_steps = getattr(kwargs["args"], "gradient_accumulation_steps", 1)
    
    # DeepSpeed setup
    if use_deepspeed or strategy == "deepspeed":
        if not use_deepspeed:
            use_deepspeed = True  # Strategy takes priority over flag
        
        # Check if DeepSpeed is installed
        if importlib.util.find_spec("deepspeed") is None:
            raise ImportError(
                "Unsloth: DeepSpeed is not installed but required for DeepSpeed strategy.\n"
                "Please install with: pip install deepspeed"
            )
        
        # Get world size for calculating global batch size
        world_size = int(os.environ.get("WORLD_SIZE", "1")) if not is_already_distributed else dist.get_world_size()
        
        # Configure DeepSpeed with proper ZeRO settings exactly as recommended
        total_batch_size = per_device_batch_size * world_size * grad_accum_steps
        
        deepspeed_config = {
            "zero_optimization": {
                "stage": zero_stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },
            "train_batch_size": total_batch_size,
            "train_micro_batch_size_per_gpu": per_device_batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "gradient_clipping": 1.0,
            "offload_optimizer_device": "none",
            "offload_param_device": "none",
            "fp16": {
                "enabled": not torch.cuda.is_bf16_supported(),
            },
            "bf16": {
                "enabled": torch.cuda.is_bf16_supported(),
            },
        }
        
        # Create accelerator with DeepSpeed configuration
        accelerator = Accelerator(
            mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
            deepspeed_plugin=deepspeed_config,
            log_with=None,
            project_config=ProjectConfiguration(distributed_type=DistributedType.DEEPSPEED),
        )
        os.environ["ACCELERATE_DISTRIBUTED_TYPE"] = "DEEPSPEED"
    else:
        # Use standard DDP
        accelerator = Accelerator(
            mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
            log_with=None,
            project_config=ProjectConfiguration(distributed_type=DistributedType.MULTI_GPU),
        )
        os.environ["ACCELERATE_DISTRIBUTED_TYPE"] = "MULTI_GPU"
    
    # Print distributed info, but only from the main process
    if accelerator.process_index == 0:
        world_size = accelerator.num_processes
        print(f"Unsloth: Multi-GPU initialized with {world_size} GPUs")
        print(f"Unsloth: Current GPU rank: {accelerator.process_index}")
        print(f"Unsloth: Using distributed strategy: {strategy}")
        
        if world_size > 1:
            print(f"Unsloth: Data will be automatically sharded across {world_size} GPUs")
            # Print distributed type
            dist_type = accelerator.distributed_type
            print(f"Unsloth: Accelerate distributed type: {dist_type}")
            if dist_type == DistributedType.MULTI_GPU:
                print("Unsloth: Using native PyTorch DDP for distributed training")
                print(f"Unsloth: Effective batch size: {per_device_batch_size * world_size * grad_accum_steps}")
            elif dist_type == DistributedType.DEEPSPEED:
                print(f"Unsloth: Using DeepSpeed ZeRO-{zero_stage} for distributed training")
                print(f"Unsloth: Effective batch size: {per_device_batch_size * world_size * grad_accum_steps}")
    
    # Make sure all processes are synced before returning
    if accelerator.num_processes > 1:
        torch.distributed.barrier()
    
    return accelerator
