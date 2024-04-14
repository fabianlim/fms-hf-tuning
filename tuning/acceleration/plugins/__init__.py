from .framework_plugin import AccelerationPlugin, AccelerationPluginInitError
from .framework_plugin_autogptq import AutoGPTQAccelerationPlugin
from .framework_plugin_bnb import BNBAccelerationPlugin

INSTALLED_PLUGINS_CLASSES = [
    AutoGPTQAccelerationPlugin,
    BNBAccelerationPlugin
]