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

# Third Party
from transformers.utils.import_utils import _is_package_available

_is_aim_available = _is_package_available("aim")
_is_fms_accelerate_available = _is_package_available("fms_acceleration")

def is_aim_available():
    return _is_aim_available

def is_fms_accelerate_available():
    return _is_fms_accelerate_available