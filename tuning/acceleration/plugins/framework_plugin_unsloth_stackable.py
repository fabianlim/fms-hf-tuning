from transformers import TrainingArguments
from peft import LoraConfig

from typing import Tuple, Dict

from .framework_plugin import AccelerationPlugin
import torch

class UnslothStackableAccelerationPlugin(AccelerationPlugin):

    require_packages = ['unsloth']
    restricted_model_archs = ['MixtralForCausalLM', 'LlamaForCausalLM', 'MistralForCausalLM']

    '''
    NOTE: unloth's LICENSE file looks like a standard Apache 2.0, but in various parts of the code, it claims to require a commercial license if used to run on more than 4 GPUs, see 
    https://github.com/unslothai/unsloth/blob/d215fd902cf28feb8abcfde2d25281d0fbf9d28c/unsloth/models/llama.py#L1140-L1143
    '''

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        self._base_layer = self._check_config_and_maybe_check_values(
            key=f"peft.quantization.unsloth.stackable.base_layer", 
            values=['auto_gptq', 'bitsandbytes']
        )
        
        # only support these at the moment
        self._check_config_equal(key=f"peft.quantization.unsloth.stackable.fused_lora", value=True)
        self._check_config_equal(key=f"peft.quantization.unsloth.stackable.fast_loss", value=True)
        self._check_config_equal(key=f"peft.quantization.unsloth.stackable.fast_rsm_layernorm", value=True)
        self._check_config_equal(key=f"peft.quantization.unsloth.stackable.fast_rope_embeddings", value=True)

    @property
    def requires_agumentation(self):
        return True

    def augmentation(
        self, 
        model, 
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        # NOTE: how do I check this now that the modifiable args are missing
        # assert peft_config.lora_dropout == 0, "Unsloth Fused Attention requires lora_dropout argument to be set to 0"

        # need to check why this is needed
        assert model.dtype == torch.float16 and train_args.fp16,\
             "need to run in fp16 mixed precision or load model in fp16"

        # guarded imports
        # - currently this function works only for auto_gptq
        from tuning.acceleration.plugin_utils.unsloth_utils import (
            add_unsloth_improvements
        )
        model = add_unsloth_improvements(model, stacked_over=self._base_layer)
        return model, modifiable_args

# register
AccelerationPlugin.register_plugin(
    UnslothStackableAccelerationPlugin,
    configuration_and_paths=["peft.quantization.unsloth.stackable"], 
)