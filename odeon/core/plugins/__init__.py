from typing import Dict, List

from odeon.core.introspection import load_instance

from .plugin import OdnPlugin

PLUGIN_REGISTRY: Dict[str, OdnPlugin] = dict()


def _register_plugin(OdnPlugin) -> None:
    PLUGIN_REGISTRY[OdnPlugin.name] = OdnPlugin


def register_plugins_elements():
    for name, plugin in PLUGIN_REGISTRY.items():
        plugin.register()


def register_plugins(plugins: List[OdnPlugin] | List[str] | str | OdnPlugin | None) -> None:
    """

    Parameters
    ----------
    plugins: List[OdnPlugin] | str | OdnPlugin | None

    Returns
    -------
    None
    """
    if isinstance(plugins, OdnPlugin):
        _register_plugin(plugins)
    elif isinstance(plugins, str):
        plugin = load_instance(plugins)
        _register_plugin(plugin)
    if isinstance(plugin, list):
        for plugin in plugin:
            if isinstance(plugin, OdnPlugin):
                _register_plugin(plugin)
            else:
                plugin = load_instance(plugin)
                _register_plugin(plugin)
    else:
        pass
