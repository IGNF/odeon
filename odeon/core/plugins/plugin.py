from abc import ABC
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from odeon.core.exceptions import MisconfigurationException
from odeon.core.introspection import load_instance
from odeon.core.logger import get_logger
from odeon.core.python_env import debug_mode
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS

logger = get_logger(logger_name=__name__, debug=debug_mode)


@unique
class PluginMaturity(str, Enum):
    """
    Enum class representing the maturity levels of a plugin.

    The PluginMaturity class provides options for specifying the maturity level
    of a plugin, including stable, experimental, development, and not specified.
    """
    STABLE = 'stable'
    EXPERIMENTAL = 'experimental'
    DEVELOPMENT = 'development'
    NOT_SPECIFIED = 'not_specified'


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True, slots=True)
class Element:
    registry: type[GenericRegistry]
    name: str
    aliases: Optional[Union[str, List[str]]]
    type_or_callable: Union[type, Callable[..., Any]]


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True, slots=True)
class Elements:
    elements: List[Element] | Dict[str, Element]

    def __iter__(self):
        if isinstance(self.elements, list):
            for element in self.elements:
                yield element
        else:
            for k, v in self.elements.items():
                yield v


class BasePlugin(ABC):
    ...


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False, slots=True)
class OdnPlugin(BasePlugin):
    name: str
    elements: Union[Element, Elements, Dict[str, PARAMS]]
    version: str = ''
    author: str = ''
    plugin_maturity: Literal[PluginMaturity.STABLE, 'stable', PluginMaturity.DEVELOPMENT, 'development',
                             PluginMaturity.EXPERIMENTAL, 'experimental',
                             PluginMaturity.NOT_SPECIFIED, 'not_specified'] = PluginMaturity.NOT_SPECIFIED
    _is_registered: bool = False

    def __post_init__(self):
        self.init()

    def init(self):
        if isinstance(self.elements, Elements):
            self.elements: Elements = self.elements
        elif isinstance(self.elements, Element):
            self.elements = Elements(elements=[self.elements])
        elif isinstance(self.elements, dict):
            l: List[Element] = list()
            for k, v in self.elements.items():
                assert isinstance(v, dict), 'your plugin has no dict params'
                # assert 'aliases' in v.keys(), f'your plugin element {str(k)} has no aliases params'
                assert 'registry' in v.keys(), f'your plugin element {str(k)} has no registry params'
                assert 'class' in v.keys(), f'your plugin element {str(k)} has no registry params'

                aliases = v['aliases'] if 'aliases' in v else None
                name = k
                registry = GenericRegistry.get(name=v['registry']) if isinstance(v['registry'], str) else v['registry']
                t = v['class']
                l.append(Element(type_or_callable=t, name=name, aliases=aliases, registry=registry))
            self.elements = Elements(elements=l)
        else:
            raise TypeError()

    def register(self, force_registry: bool = False):
        if self._is_registered is False or force_registry:
            for element in self.elements:
                try:
                    registry = element.registry
                    registry.register_element(t=element.type_or_callable, name=element.name)
                    registry.register_aliases(name=element.name, aliases=element.aliases)
                except KeyError as e:
                    raise MisconfigurationException(message=f'something went wrong during plugin configuration,'
                                                            f'it seems like your plugin name or one of your alias is '
                                                            f'already exists in registry {str(element.registry)},'
                                                            f' details of the error: {str(e)}')
            self._is_registered = True


def _register_plugin(plugin: str | OdnPlugin, force_registry: bool = False):
    if isinstance(plugin, str):
        instance = load_instance(plugin)
        try:
            assert isinstance(instance, OdnPlugin)
        except AssertionError:
            logger.error(f'plugin {instance} is not an instance of OdnPlugin')
        instance.register(force_registry=force_registry)
    else:
        try:
            assert isinstance(plugin, OdnPlugin)
        except AssertionError:
            logger.error(f'plugin {plugin} is not an instance of OdnPlugin')

        plugin.register(force_registry=force_registry)


def load_plugins(plugins: Dict[str, OdnPlugin | str] | List[OdnPlugin | str] | str | OdnPlugin | None,
                 force_registry: bool = False) -> None:
    logger.debug(f'plugins will be loaded: {plugins}')
    if isinstance(plugins, list):
        for plugin in plugins:
            _register_plugin(plugin=plugin, force_registry=force_registry)
    if isinstance(plugins, dict):
        logger.debug(f'plugins are loaded as dict: {plugins}')
        for _, plugin in plugins.items():
            _register_plugin(plugin=plugin, force_registry=force_registry)
    elif isinstance(plugins, OdnPlugin):
        _register_plugin(plugin=plugins, force_registry=force_registry)
    elif isinstance(plugins, str):
        _register_plugin(plugin=plugins, force_registry=force_registry)
    else:
        logger.debug(f'plugin or plugins are not from a supported type: {type(plugins)}')
