
from transformers.utils.import_utils import _is_package_available
from peft import LoraConfig
from peft.tuners.lora.gptq import QuantLinear as LoraLinearGPTQ

import torch

def replace_module_peft(self, parent_module, child_name, new_module, old_module):

    # replace the lora linear
    setattr(parent_module, child_name, new_module)

    # dispatch to correct device 
    # FIXME: refactor
    for name, module in new_module.named_modules():
        if "lora_" in name:
            device = (list(old_module.parameters()) + list(old_module.buffers()))[0].device
            module.to(device)



def create_new_module_peft(
    lora_config: LoraConfig,
    adapter_name: str,
    target: torch.nn.Module,
    target_cls,
    **kwargs,
):
    # if the base layer module matches a supported class, dispatch the lora linear
    # to be installed
    new_module = None
    if isinstance(target, target_cls):
        new_module = LoraLinearGPTQ(target, adapter_name, lora_config=lora_config, **kwargs)

    # if module cannot be found, return None which results in a raise in the call-stack
    return new_module