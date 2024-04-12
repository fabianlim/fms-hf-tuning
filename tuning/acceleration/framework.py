
def parse_configuration():
    pass


from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training

from types import MethodType

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.utils.peft_utils import get_gptq_peft_model
from auto_gptq.utils.peft_utils import GPTQLoraModel
from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear as triton_Qlinear

from peft.tuners.lora.gptq import QuantLinear as LoraLinearGPTQ
from peft.tuners.lora.model import LoraModel

from typing import Tuple

import torch

def _replace_module(self, parent_module, child_name, new_module, old_module):

    # replace the lora linear
    setattr(parent_module, child_name, new_module)

    # dispatch to correct device 
    # FIXME: refactor
    for name, module in new_module.named_modules():
        if "lora_" in name:
            device = (list(old_module.parameters()) + list(old_module.buffers()))[0].device
            module.to(device)

# def _dispatch_lora_linear_for_gptq(
def _create_new_module_triton(
    lora_config: LoraConfig,
    adapter_name: str,
    target: torch.nn.Module,
    **kwargs,
):
    # if the base layer module matches a supported class, dispatch the lora linear
    # to be installed
    new_module = None
    if isinstance(target, triton_Qlinear):
        new_module = LoraLinearGPTQ(target, adapter_name, lora_config=lora_config, **kwargs)

    # if module cannot be found, return None which results in a raise in the call-stack
    return new_module

class AccelerationFramework:
    def __init__(self, configuration_file: str=None):

        # Currently we only support the triton_v2 kernel, so there is nothing
        # much else to configure
        pass

    def model_loader(self, model_name: str, **kwargs):

        # Currently we allow only a quantized checkpoint to be loaded, we do not
        # implement the quantization process here. 
        #
        # The quantization process is used to convert a non-quantized checkpoint
        # (provided in model_name) into a quantized one. This entails
        # 1. providing a BaseQuantizeConfig with the appropriate quantization settings
        # 2. calling BaseGPTQForCausalLM.quantize to run the quantization algorithm (may take time, e.g. hours)
        # 3. calling BaseGPTQForCausalLM.save_pretrained to save a quantized checkpoint
        # 
        # The reasons for not implementing the flow at this point are.
        # 1. The quantization can take very long for large models. As such, it is more appropriate
        #    to run it once outside of training, and save the checkpoint to be used for multiple runs.
        # 2. Requires some API changes to point to where the quantized checkpoint should be saved.
        #    Can be confusing to the user since it will be different from model_name

        # NOTE: there will be a warning that can be ignored
        # "WARNING - QuantLinear with the exllama backend not does support the trainable mode yet, switching to cuda/cuda_old/triton backend."

        # assume model_name points to a quantized checkpoint. Thus we load the quantization
        # config directly from the checkpoint.
        quantize_config = BaseQuantizeConfig.from_pretrained(model_name)

        # get additional parameters
        torch_dtype = kwargs.get('torch_dtype', torch.float32)
        low_cpu_mem_usage = kwargs.get('low_cpu_mem_usage', False)

        model = AutoGPTQForCausalLM.from_quantized(
            model_name, 
            quantize_config=quantize_config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_marlin=False, # disable, cannot be used for training (no forward+backward)
            disable_exllama=True, # disable, cannot be used for training (no backward)
            warmup_triton=False, # disable for now, because it will try to run the warmup while on CPU
            use_tritonv2=True,
            trainable=True, # only support trainable mode
        )

        # these will be properly set since it is not loaded using from_pretrained
        # - so, set them here. 
        # - in particular "is_loaded_in_4bit" will be checked in prepare_model_for_kbit_training
        #   and there is a section of code that will be skipped if not set.
        setattr(model, "is_loaded_in_4bit", True)
        setattr(model, "quantization_method", 'gptq')
        return model

    @property
    def requires_custom_loading(self):
        return True

    @property
    def requires_agumentation(self):
        return True

    def augmentation(
        self, 
        model, 
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        peft_config, = modifiable_args # unpack modifiable args

        # some assertions
        assert peft_config is not None, "need peft_config to install PEFT adapters"
        assert model.dtype == torch.float16 or train_args.fp16,\
             "need to run in fp16 mixed precision or load model in fp16"

        # call the prepare_model_for_kbit_training. This will no longer be called
        # inside SFTTrainer, because we eventually return None for the peft_config.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=train_args.gradient_checkpointing_kwargs,
        )

        # These functions need to replaced due to some incompatibliites 
        # with newer PEFT packages.
        # - on augmentation we call auto_gptq.utils.peft_utils.get_gptq_peft_model
        # - this internally calls peft.utils.other.get_peft_model
        # - however the problem is that peft API moves very fast, and there are incompatiblities
        # 
        # During peft wrapping there are two key operations
        # 1. LoraModel._create_new_module is called to create a LoraLinear layer that is
        #    compatible with the base layer. For quantized base layers, the LoraLinear
        #    may be different.
        # 2. GPTQLoraModel._replace_module to replace the existing Linear with the LoraLinear.
        #    Also move to device (which may depend on how base layer is implemented)

        # NOTE: GPTQLoraModel inherits from LoraModel, and the _create_new_module method is called
        # on the parent. Hence _create_new_module is patched on the parent

        # FIXME: 
        # 1. investigate using BaseGPTQForCausalLM.make_sure_compatible_with_peft
        #    to see if we can get around the patching

        _old_create_new_module = LoraModel
        _old_replace_module = GPTQLoraModel._replace_module
        LoraModel._create_new_module = staticmethod(_create_new_module_triton)
        GPTQLoraModel._replace_module = MethodType(_replace_module, GPTQLoraModel)

        # Install GPTQ adapters using the AutoGPTQ package (with the above patches)
        model = get_gptq_peft_model(
            model, 
            peft_config=peft_config, 
            auto_find_all_linears=peft_config.target_modules is None,
            train_mode=True, # install adapaters for training
        )
        modifiable_args = (None, ) # return a None for peft_config

        # undo the patching for hygine
        LoraModel._create_new_module = staticmethod(_old_create_new_module)
        GPTQLoraModel._replace_module = MethodType(_old_replace_module, GPTQLoraModel)

        return model, modifiable_args