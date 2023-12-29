from typing import Callable

from odeon.core.registry import GenericRegistry
from odeon.core.plugins.plugin import Element


def my_dummy_function() -> bool:
    return True


def test_valid_instance_creation():
    registry = GenericRegistry[Callable]
    name = "example_element"
    aliases = ["alias1", "alias2"]
    t = my_dummy_function
    element = Element(registry=registry, name=name, aliases=aliases,
                      type_or_callable=t)

    assert element.registry == registry
    assert element.name == name
    assert element.aliases == aliases
    assert element.type_or_callable == t
