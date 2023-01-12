from typing import Callable, Dict, List, Type, Union

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS


@GenericRegistry.register('transform', aliases=['trans'])
class TransformRegistry(GenericRegistry[Type[Callable]]):
    @classmethod
    def register_fn(cls, cl: Callable, name: str):
        assert callable(cl)
        cls._registry[name] = cl


def build_transform(transforms: List[Union[Dict, Callable]] | Dict[str, PARAMS]) -> List[Callable]:
    result: List[Callable] = list()
    if isinstance(transforms, list):
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
    elif isinstance(transforms, dict):
        for key, value in transforms.items():
            if value is not None:
                result.append(TransformRegistry.create(name=key, **value))
            else:
                result.append(TransformRegistry.create(name=key))
    else:
        raise MisconfigurationException(message=f'tranforms params {transforms} is not a list or dict but'
                                                f'has type {str(type(transforms))}')
    return result
