from .plugins.plugin import OdnPlugin, PluginMaturity
from .app import APP_REGISTRY
from .registry import GenericRegistry

app_plugin = OdnPlugin(name='app_plugin',
                       author='samy KHELIFI-RICHARDS',
                       plugin_maturity=str(PluginMaturity.DEVELOPMENT.value),
                       version='0.1',
                       elements={'app_registry': {'class': APP_REGISTRY,
                                 'registry': GenericRegistry}}
                       )
