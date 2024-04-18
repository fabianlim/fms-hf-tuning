from .framework_plugin import AccelerationPlugin, get_relevant_configuration_sections

# can this be automated?
from .framework_plugin_autogptq import AutoGPTQAccelerationPlugin
from .framework_plugin_bnb import BNBAccelerationPlugin

try:
    from .framework_plugin_unsloth_autogptq import UnslothAutoGPTQAccelerationPlugin
except ImportError:
    pass # might fail due to missing xformers