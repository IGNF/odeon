from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS


class PluginMaturity(str, Enum):
    STABLE = 'stable'
    EXPERIMENTAL = 'experimental'
    DEVELOPMENT = 'development'
    NOT_SPECIFIED = 'not_specified'


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True, slots=True)
class Element:
    registry: GenericRegistry
    name: str
    aliases: Optional[Union[str, List[str]]]
    cl: type


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
                cl = v['class']
                l.append(Element(cl=cl, name=name, aliases=aliases, registry=registry))
            self.elements = Elements(elements=l)
        else:
            raise TypeError()

    def register(self):
        for element in self.elements:
            try:
                registry = element.registry
                registry.register_class(cl=element.cl, name=element.name)
                registry.register_aliases(name=element.name, aliases=element.aliases)
            except KeyError as e:
                raise MisconfigurationException(message=f'something went wrong during plugin configuration,'
                                                        f'it seems like your plugin name or one of your alias is '
                                                        f'already exists in registry {str(element.registry)},'
                                                        f' details of the error: {str(e)}')
