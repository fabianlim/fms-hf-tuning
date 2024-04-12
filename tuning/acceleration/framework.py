
def parse_configuration():
    pass

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.utils.peft_utils import get_gptq_peft_model

from transformers import TrainingArguments
from peft import LoraConfig

from types import MethodType
from auto_gptq.utils.peft_utils import GPTQLoraModel
from peft.tuners.lora.model import LoraModel

import torch

def _replace_module(self, parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    # dispatch to correct device
    for name, module in new_module.named_modules():
        if "lora_" in name:
            device = (list(old_module.parameters()) + list(old_module.buffers()))[0].device
            module.to(device)

from peft.tuners.lora.gptq import QuantLinear as PEFT_LoraLinear

def _dispatch_gptq_triton(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs,
):
    new_module = None
    from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear as triton_Qlinear
    if isinstance(target, triton_Qlinear):
        new_module = PEFT_LoraLinear(target, adapter_name, **kwargs)
    return new_module

# refactor!
def _create_new_module_triton(lora_config, adapter_name, target, **kwargs):
    return _dispatch_gptq_triton(target, adapter_name, lora_config=lora_config, **kwargs)

class AccelerationFramework:
    def __init__(self, configuration_file: str=None):
        pass

    def model_loader(self, model_name: str, **kwargs):
        quantize_config = BaseQuantizeConfig.from_pretrained(model_name)
        LoraModel._create_new_module = staticmethod(_create_new_module_triton)
        GPTQLoraModel._replace_module = MethodType(_replace_module, GPTQLoraModel)

        torch_dtype = kwargs.get('torch_dtype', torch.float32)

        return AutoGPTQForCausalLM.from_quantized(
            model_name, quantize_config = quantize_config,
            torch_dtype=torch_dtype,
            use_marlin = False,
            disable_exllama = True,
            # warmup_triton = True,
            warmup_triton= False, # disable for now, because it will try to run the warmup while on CPU
            use_tritonv2 = True,
            trainable = True,
        )

    @property
    def requires_custom_loading(self):
        return True

    @property
    def requires_agumentation(self):
        return True

    def augmentation(
        self, 
        model, 
        accelerator,
        train_args: TrainingArguments, 
        peft_config: LoraConfig,
    ):
        assert peft_config is not None, "need peft_config to install PEFT adapters"

        # FIXME: handle this more properly. Need to check if torch_dtype was specified as fp16 also
        # assert train_args.fp16, "need to run in fp16 mixed precision or load model in fp16"

        # this is a hack to enter the enable_grads part of the code in 
        # prepare_model_for_kbit_training below

        setattr(model, "is_loaded_in_4bit", True)
        setattr(model, "quantization_method", 'gptq')

        # we need to call this here, because it is supposed to be called inside 
        # SFTTRainer, but because we pass a already PEFTed model into Trainer, 
        # it will skip that part of the code
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=train_args.gradient_checkpointing_kwargs,
        )

        # PEFT Installation
        return get_gptq_peft_model(
            model, 
            peft_config = peft_config, 
            auto_find_all_linears=False,
            train_mode = True,
        )