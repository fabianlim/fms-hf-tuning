import torch
from typing import List, Dict, Tuple, Optional, Set
from transformers import TrainingArguments
from peft import LoraConfig

class AccelerationPlugin:

    configuration_keys: List[str] = []
    restricted_model_archs: Optional[Set] = None
    require_packages: Optional[Set] = None

    def __init__(self, configurations: List[Dict]):

        # will pass in a list of dictionaries keyed by "configuration_keys"
        # to be used for initialization
        pass

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