from .framework_plugin import AccelerationPlugin, AccelerationPluginInitError
from .framework_plugin_autogptq import AutoGPTQAccelerationPlugin

INSTALLED_PLUGINS_CLASSES = [
    AutoGPTQAccelerationPlugin
]