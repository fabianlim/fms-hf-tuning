
def parse_configuration():
    pass

import torch
from transformers import TrainingArguments
from peft import LoraConfig
from peft.tuners.lora.model import LoraModel
from auto_gptq.utils.peft_utils import GPTQLoraModel
from unsloth import FastLanguageModel
from unsloth.utils.modeling import QuantizationMethod
from unsloth.gptq.triton.layers import GPTQuantLinear

from types import MethodType
from typing import Tuple, List, Dict

from .framework_plugin import AccelerationPlugin
from .framework_plugin_autogptq import _replace_module, LoraLinearGPTQ

# def _dispatch_lora_linear_for_gptq(
def _create_new_module_triton(
    lora_config: LoraConfig,
    adapter_name: str,
    target: torch.nn.Module,
    **kwargs,
):
    # if the base layer module matches a supported class, dispatch the lora linear
    # to be installed
    new_module = target
    if (
        isinstance(target, GPTQuantLinear)
    ):
        new_module = LoraLinearGPTQ(target, adapter_name, **kwargs)
    # if module cannot be found, return None which results in a raise in the call-stack
    return new_module

class UnslothAutoGPTQAccelerationPlugin(AccelerationPlugin):
    
    require_packages = ['auto_gptq', 'unsloth', 'optimum']
    restricted_model_archs = ['MixtralForCausalLM', 'LlamaForCausalLM', 'MistralForCausalLM', 'GemmaForCausalLM']

    '''
    NOTE:
    Unsloth's OSS license allows for up to 4 GPUs, will require a commercial license for larger resources
    https://github.com/unslothai/unsloth/blob/d215fd902cf28feb8abcfde2d25281d0fbf9d28c/unsloth/models/llama.py#L1140-L1143
    '''

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # just do checking, nothing must to configure at this point
        # if need to configure then do something like this:
        # self.kernel = self._get_config_value("peft.kernel")
        self._check_config_equal(key="peft.quantization", value="unsloth")
        self._check_config_equal(key="peft.quantization.unsloth.base_layer", value="auto_gptq")
        self._check_config_equal(key="peft.quantization.unsloth.kernel", value="triton_v2")

    def model_loader(self, model_name: str, **kwargs):
        # 1. Load the gptq base model through unsloth FastLanguageModel
        torch_dtype = kwargs.get('torch_dtype', torch.float16)
        model, _ = FastLanguageModel.from_pretrained(
                model_name,
                dtype=torch_dtype,
                device_map="auto",
                quantization_method=QuantizationMethod.GPTQ,
            )

        # 2. Depending on the model, replace appropriate CUDA kernel with TritonV2 kernel 
        # - `FastLanguageModel` above will load a "default" cuda linear kernel. 
        # - the idea is to load the appropriate "default" kernel (for each model), then follow up with an injection
        #.  via the `inject_to_model(..., target_module_type=DefaultKernel)` method.
        if model.config.model_type == 'mixtral':
            from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
                QuantLinear as QuantLinearCuda,
            )                
        else:
            from auto_gptq.nn_modules.qlinear.qlinear_cuda import (
                QuantLinear as QuantLinearCuda,
            )
        GPTQuantLinear.inject_to_model(
            model, target_module_type=QuantLinearCuda
        )

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
        assert peft_config.lora_dropout == 0, "Unsloth Fused Attention requires lora_dropout argument to be set to 0"
        assert model.dtype == torch.float16 or train_args.fp16,\
             "need to run in fp16 mixed precision or load model in fp16"

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

        # In the unsloth implementation, the prepare_model_for_kbit_training get_peft_model is called 
        # inside `FastLanguageModel.get_peft_model`
        if model.config.model_type == 'mixtral':
            from unsloth.models.mixtral import FastMixtralModel
            _lib = FastMixtralModel
        else:
            _lib = FastLanguageModel
      
        model = _lib.get_peft_model(
            model,
            use_gradient_checkpointing=train_args.gradient_checkpointing,
            **{k:v for k,v in peft_config.to_dict().items() if k != 'task_type'},
        )
        modifiable_args = (None, ) # return a None for peft_config

        # undo the patching for hygine
        LoraModel._create_new_module = staticmethod(_old_create_new_module)
        GPTQLoraModel._replace_module = MethodType(_old_replace_module, GPTQLoraModel)

        return model, modifiable_args

# register
AccelerationPlugin.register_plugin(
    UnslothAutoGPTQAccelerationPlugin,
    configuration_and_paths=["peft.quantization.unsloth"], 
)