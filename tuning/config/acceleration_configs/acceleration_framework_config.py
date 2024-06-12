# Copyright The FMS HF Tuning Authors
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

from dataclasses import dataclass, fields, asdict, is_dataclass
from typing import Annotated, List, Dict, Type
from tuning.utils.import_utils import is_fms_accelerate_available
from .quantized_lora_config import AutoGPTQLoraConfig, BNBQLoraConfig
import yaml

if is_fms_accelerate_available():
    # Third Party
    from fms_acceleration import AccelerationFramework  # pylint: disable=import-error

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
@dataclass
class AccelerationFrameworkConfig:
    "Dataclass that manages configuration of AccelerationFramework"

    # each field will a single-level use case dataclass
    auto_gptq: Annotated[AutoGPTQLoraConfig, "peft", "quantization"] = None

    bitsandbytes: Annotated[BNBQLoraConfig, "peft", "quantization"] = None

    @staticmethod
    def from_dataclasses(*dataclasses: Type):
        "Convert one or many FMS config dataclasses to a monolithic AccelerationConfig"

        config = AccelerationFrameworkConfig()
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

    def get_framework(self):

        if is_fms_accelerate_available():

            # to be eventually be made to be passed as a dict to Acceleration
            # Framework
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile('w') as f:
                self.to_yaml(f.name)
                return AccelerationFramework(f.name)
        else:
            raise ValueError(
                "Specified acceleration framework configs "
                "but fms_acceleration package not available"
            )

    def to_dict(self):
        """convert a valid AccelerationFrameworkConfig dataclass into a schema-less dictionary
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
        for fi in fields(self):
            datacls = getattr(self, fi.name)
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

    def to_yaml(self, filename: str, top_level_key: str = 'plugins'):
        "convert a valid AccelerationConfig dataclass into a yaml"
        configuration_contents = self.to_dict()
        with open(filename, "w") as f:
            yaml.dump({top_level_key: configuration_contents}, f)
