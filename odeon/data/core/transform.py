from typing import Callable, Dict, List, Type, Union

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry


@GenericRegistry.register('transform', aliases=['trans'])
class TransformRegistry(GenericRegistry[Type[Callable]]):
    @classmethod
    def register_fn(cls, cl: Callable, name: str):
        assert callable(cl)
        cls._registry[name] = cl


def build_transform(transforms: List[Union[Dict, Callable]]) -> List[Callable]:
    result: List[Callable] = list()
    for transform in transforms:
        if isinstance(transform, dict):
            name = transform['name']
            if 'params' in transform:
                params: Dict = transform['params']
                result.append(TransformRegistry.create(name=name, **params))
            else:
                result.append(TransformRegistry.create(name=name))
        elif callable(transform):
            result.append(transform)
        else:
            raise MisconfigurationException(message=f'transform param {transform} is neither a dict or a callable')
    return result
