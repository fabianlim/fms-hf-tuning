
def parse_configuration():
    pass


from transformers import BitsAndBytesConfig, AutoModelForCausalLM

from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from typing import Dict, Tuple

from .framework_plugin import AccelerationPlugin

import torch
import inspect, warnings

# this is a modified copy of the function from peft.utils.other, that we 
# will instead use 
# - in the original version, all non-INIT8 params (e.g., fp16, bf16) are upcast
#   to full precision.
# - this will cause problems in the LoraLayers, because the activations will then
#   be constantly downcasted, resulting in greatly reduced throughput.
def _prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # To support older transformers versions, check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored."
                " if you want to use that feature, please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {} if not _supports_gc_kwargs else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model


class BNBAccelerationPlugin(AccelerationPlugin):
    
    require_packages = ['bitsandbytes']

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        self._quant_type = self._check_config_and_maybe_check_values(
            key="peft.quantization.bitsandbytes.quant_type", 
            values=["fp4", "nf4"]
        )

    def model_loader(self, model_name: str, **kwargs):

        # get additional parameters
        torch_dtype = kwargs.get('torch_dtype', torch.float32)
        # low_cpu_mem_usage = kwargs.get('low_cpu_mem_usage', False)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = self._quant_type,
            bnb_4bit_compute_dtype    = torch_dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = torch_dtype,
            quantization_config = bnb_config,
            token = None,
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
        assert model.dtype == torch.float16 or train_args.fp16,\
             "need to run in fp16 mixed precision or load model in fp16"

        # requires a custom prepare because the stock one in peft will introduce
        # extraneous casting
        model = _prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=train_args.gradient_checkpointing_kwargs,
        )

        model = get_peft_model(model, peft_config)
        modifiable_args = (None, ) # return a None
        return model, modifiable_args

# register
AccelerationPlugin.register_plugin(
    BNBAccelerationPlugin,
    configuration_and_paths=["peft.quantization.bitsandbytes"], 
)