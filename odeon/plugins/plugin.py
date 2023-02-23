from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True, slots=True)
class Element:
    registry: GenericRegistry
    name: str
    aliases: Optional[Union[str, List[str]]]


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


class OdnPlugin(ABC):
    ...


class Plugin(OdnPlugin):

    def __init__(self, elements: Union[Element, Elements, Dict[str, PARAMS]]):
        super().__init__()
        assert isinstance(elements, Elements) or isinstance(elements, Element)
        if isinstance(elements, Elements):
            self.elements: Elements = elements
        elif isinstance(elements, Element):
            self.elements: Elements = Elements(elements=[elements])
        elif isinstance(elements, dict):
            l: List[Element] = list()
            for k, v in elements.items():
                assert isinstance(v, dict), 'your plugin has no dict params'
                assert 'aliases' in v.keys(), f'your plugin element {str(k)} has no aliases params'
                assert 'registry' in v.keys(), f'your plugin element {str(k)} has no registry params'
                aliases = v['aliases']
                name = k
                registry = GenericRegistry.get(name=v['registry'])
                l.append(Element(name=name, aliases=aliases, registry=registry))
            self.elements: Elements = Elements(elements=l)

        else:
            raise TypeError()

    def register(self):
        for element in self.elements:
            try:
                registry = element.registry
                registry.register_class(name=element.name, aliases=element.aliases)
            except KeyError as e:
                raise MisconfigurationException(message=f'something went wrong during plugin configuration,'
                                                        f'it seems like your plugin name or one of your alias is '
                                                        f'already exists in registry {str(element.registry)},'
                                                        f' details of the error: {str(e)}')
