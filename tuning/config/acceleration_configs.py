from dataclasses import dataclass, fields, asdict, is_dataclass
from typing import Annotated, List, Dict, Type
from tuning.utils.import_utils import is_fms_accelerate_available
import yaml

# DESIGN OF FMS CONFIGS:
# - FMS will have differnt configs (probably one (or more) / plugin).
# - e,g. QuantizedLoraConfig will be for the accelerated_peft plugin
# - e.g, FusedOpsAndKernelsConfig will be for fused_ops_and_kernels plugin
# - FMS users will understand that to use thse configs, they will need
#   to install the plugin that corresponds to that config
# - each FMS config will nest multiple dataclasses in a single level
# - typically each nested dataclass corresponds to one use case
# - e.g. for the QuantizedLoraConfig, two use cases of auto_gptq and bnb_qlora

# - the HF dataclass argument parser will create position arguments from the
#   FMS config
# - in the usal way, the keys of the FMS config will correspond to a --key
# - then the use case dataclass will be passed its attributes by position
# - hence, this is the reason why we enforce the FMS config to be
#   single-level nested dataclasses.

@dataclass
class QuantizedLoraConfig:

    # to use auto_gptq 4bit lora base layers
    auto_gptq: "AutoGPTQLoraConfig" = None

    # to use auto_gptq 4bit lora base layers
    bnb_qlora: "BNBQLoraConfig" = None

    def __post_init__(self):
        if self.auto_gptq is None and self.bnb_qlora is None:
            raise ValueError('at least one quantized config has to be specified.')
    

@dataclass
class AutoGPTQLoraConfig:

    # auto_gptq supports various kernels, to select the kernel to use.
    kernel: str = 'triton_v2'

    # allow auto_gptq to quantize a model before training commences.
    # NOTE: currently this is not allowed.
    from_quantized: bool = True

    def __post_init__(self):
        
        if self.kernel != 'triton_v2':
            raise ValueError("only 'triton_v2' kernel currently supported.")

        if not self.from_quantized:
            raise ValueError("only 'from_quantized' == True currently supported.")

@dataclass
class BNBQLoraConfig:

    # type of quantization applied
    quant_type: str = 'nf4'

    # if we only want to quantize the base layer, and defer to the 
    # huggingface to prepare the peft (i.e. lora) model
    no_peft_model: bool = False

    def __post_init__(self):
        if self.quant_type not in ['nf4', 'fp4']:
            raise ValueError("quant_type can only be either 'nf4' or 'fp4.")

if is_fms_accelerate_available():

    # DESIGN OF ACCELERATION CONFIGS
    # - An ACCELERATION CONFIG is a monolothic config passed to AccelerationFramework
    # - it is NOT meant to be user facing. Users will only configure
    #   use case dataclasses within.
    # - however, uses can consult the annotations (see below) to understand
    #   which use-case config can be active at the same time.
    # - it is a collection of use-case dataclasses (see above)
    # - every use-case dataclass is annotated with a header
    # - any two use-case dataclasses that are annotated with the 
    #   same header, cannot be active at the same time.
    # - An Acceleration Config is valid only if it does not have any 
    #   use-case dataclass that violates these rules.
    from fms_acceleration.constants import KEY_PLUGINS

    def parse_acceleration_config(config: "AccelerationConfig"):
        """convert a valid AcceleratoinConfig dataclass into a schema-less dictionary
        as dictated by the header annotations.
        """

        # populate a dictionary
        configuration_contents = {}

        # helper function to populate
        def _descend_and_set(path: List[str], d: Dict):
            r = configuration_contents
            for p in path[:-1]:
                if p not in r:
                    r[p] = {} # branch
                r = r[p]

            r[path[-1]] = d

        # parse each field
        already_set = set()
        for fi in fields(config):
            datacls = getattr(config, fi.name)
            if datacls:
                # this is the documented way to get annotations
                # https://docs.python.org/3/library/typing.html#typing.Annotated
                prefix_path = fi.type.__metadata__
                if prefix_path in already_set:
                    raise ValueError(f"configuration path '{prefix_path}' already occupied.")

                path = prefix_path + (fi.name,)
                already_set.add(prefix_path)
                _descend_and_set(path, asdict(datacls))

        return configuration_contents

    def parse_acceleration_config_to_yaml(config: "AccelerationConfig", filename: str):
        "convert a valid AccelerationConfig dataclass into a yaml"
        configuration_contents = parse_acceleration_config(config)
        with open(filename, "w") as f:
            yaml.dump({KEY_PLUGINS: configuration_contents}, f)

    @dataclass
    class AccelerationConfig:
        "Dataclass that manages configuration of AccelerationFramework"

        # each field will a single-level dataclass
        auto_gptq: Annotated[AutoGPTQLoraConfig, "peft", "quantization"] = None

        bitsandbytes: Annotated[BNBQLoraConfig, "peft", "quantization"] = None

    def convert_dataclasses_to_acceleration_config(*dataclasses: Type):
        "Convert one or many FMS configs to a monolithic AccelerationConfig"

        config = AccelerationConfig()
        rem_fields = {fi.name: fi for fi in fields(config)} # these need to be parsed

        def _convert(*dcs: Type):
            for dc in dcs:
                nested_dataclasses = []
                for fi in fields(dc):
                    attr = getattr(dc, fi.name) 
                    if is_dataclass(attr):
                        nested_dataclasses.append(attr)
                
                if len(nested_dataclasses) > 0:
                    _convert(*nested_dataclasses)
                    return

                # otherwise it must be a pure dataclass by design
                found = set()
                for fi in rem_fields.values():
                    if isinstance(dc, fi.type.__origin__):
                        setattr(config, fi.name, dc)
                        found.add(fi.name)
                for name in found:
                    del rem_fields[name]

        _convert(*dataclasses)
        return config

