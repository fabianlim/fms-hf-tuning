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

from typing import Type

from dataclasses import fields
from transformers.hf_argparser import string_to_bool, DataClass

def ensure_nested_dataclasses_initialized(dataclass: DataClass):
    for f in fields(dataclass):
        nested_type = f.type
        values = getattr(dataclass, f.name)
        if values is not None:
            values = nested_type(*values)
        setattr(dataclass, f.name, values)

class EnsureTypes:

    def __init__(self, *types: Type):
        map = {bool: string_to_bool}
        self.types = [map.get(t, t) for t in types]
        self.reset()

    def reset(self):
        self.cnt = 0

    def __call__(self, val):
        if self.cnt >= len(self.types):
            raise ValueError(
                "EnsureTypes require 'reset' to be called to be re-used."
            )

        t = self.types[self.cnt]
        self.cnt += 1
        return t(val)
