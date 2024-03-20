from .app import APP_REGISTRY
from .plugins.plugin import OdnPlugin, PluginMaturity
from .registry import GENERIC_REGISTRY

app_plugin = OdnPlugin(name='app_plugin',
                       author='samy KHELIFI-RICHARDS',
                       plugin_maturity=PluginMaturity.DEVELOPMENT.value,
                       version='0.1',
                       elements={'app_registry': {'class': APP_REGISTRY,
                                 'registry': GENERIC_REGISTRY}}
                       )
