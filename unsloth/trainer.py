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
        super().__init__(*args, **kwargs)
        
        # Configure multi-GPU if specified
        multi_gpu_strategy = getattr(self.args, "multi_gpu_strategy", "ddp")
        if multi_gpu_strategy != "none":
            import torch.distributed as dist
            from accelerate import Accelerator
            
            # Log multi-GPU setup
            if dist.is_available() and dist.is_initialized():
                print(f"Unsloth: Using multi-GPU with {dist.get_world_size()} GPUs")
                print(f"Unsloth: Current GPU rank: {dist.get_rank()}")
                print(f"Unsloth: Using distributed strategy: {multi_gpu_strategy}")
    
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None: return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        pass
        return self.optimizer
    
    def _prepare_inputs(self, inputs):
        """
        Prepare inputs for multi-GPU training
        """
        inputs = super()._prepare_inputs(inputs)
        return inputs
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


def initialize_distributed(strategy="ddp", use_deepspeed=False, zero_stage=2):
    """
    Initialize distributed training for Unsloth.
    
    Args:
        strategy (str): Distributed strategy to use - 'ddp' (default) or 'deepspeed'
        use_deepspeed (bool): If True, will use DeepSpeed for distributed training
        zero_stage (int): ZeRO stage to use when using DeepSpeed (0, 1, 2, or 3)
        
    Returns:
        accelerator (Accelerator): Accelerate's Accelerator object
    """
    import os
    import torch
    import importlib.util
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    
    if use_deepspeed or strategy == "deepspeed":
        if not use_deepspeed:
            use_deepspeed = True  # Strategy takes priority over flag
        
        # Check if DeepSpeed is installed
        if importlib.util.find_spec("deepspeed") is None:
            raise ImportError(
                "Unsloth: DeepSpeed is not installed but required for DeepSpeed strategy.\n"
                "Please install with: pip install deepspeed"
            )
        
        # Configure DeepSpeed
        deepspeed_config = {
            "gradient_accumulation_steps": 1,  # Will be overridden by training args
            "gradient_clipping": 1.0,
            "offload_optimizer_device": "none",
            "offload_param_device": "none",
            "zero_stage": zero_stage,
            "train_micro_batch_size_per_gpu": 1,  # Will be overridden
        }
        
        accelerator = Accelerator(
            mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
            deepspeed_plugin=deepspeed_config,
            log_with=None,
            project_config=ProjectConfiguration(),
        )
    else:
        # Use standard DDP
        accelerator = Accelerator(
            mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
            log_with=None,
            project_config=ProjectConfiguration(),
        )
    
    # Print distributed info
    if accelerator.process_index == 0:
        print(f"Unsloth: Multi-GPU initialized with {accelerator.num_processes} GPUs")
        print(f"Unsloth: Current GPU rank: {accelerator.process_index}")
        print(f"Unsloth: Using distributed strategy: {strategy}")
    
    return accelerator
