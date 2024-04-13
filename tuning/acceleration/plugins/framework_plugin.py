import torch
from typing import List, Dict, Tuple, Optional, Set, Any
from transformers import TrainingArguments
from peft import LoraConfig

class AccelerationPlugin:

    configuration_keys: List[str] = []
    restricted_model_archs: Optional[Set] = None
    require_packages: Optional[Set] = None

    def __init__(self, configurations: Dict[str, Dict]):

        # will pass in a list of dictionaries keyed by "configuration_keys"
        # to be used for initialization
        self.configurations = configurations

    @property
    def requires_custom_loading(self):
        return False

    @property
    def requires_agumentation(self):
        return False

    def model_loader(self, model_name: str, **kwargs):
        raise NotImplementedError

    def augmentation(
        self, 
        model: torch.nn.Module, 
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        raise NotImplementedError

    def callbacks(self):
        return []

    def _get_config_value(self, key: str):
        t = self.configurations
        for k in key.split('.'):
            t = t[k]
        return t

    def _check_config_in_values(
        self, key: str, values: List[Any], message: str = None
    ):
        t = self.configurations
        for k in key.split('.'):
            t = t[k]
        if t not in values:
            raise AccelerationPluginInitError(
                message if message is not None else
                f"\'{key}\' must be in \'{values}\'"
            )

    def _check_config_equal(self, key: str, value: Any, **kwargs):
        return self._check_config_in_values(key, [value], **kwargs)

class AccelerationPluginInitError(Exception):
    pass
