from typing import Callable, Dict, List, Union

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS

TRANSFORM_REGISTRY = GenericRegistry[Callable]


def build_transform(transforms: List[Union[Dict, Callable]] | Dict[str, PARAMS]) -> List[Callable]:
    result: List[Callable] = list()
    if isinstance(transforms, list):
        for transform in transforms:
            if isinstance(transform, dict):
                name = transform['name']
                if 'params' in transform:
                    params: Dict = transform['params']
                    instance = TRANSFORM_REGISTRY.create(name=name, **params)
                    assert instance is not None, f'expected Callable, got None for transform {instance}'
                    result.append(instance)
                else:
                    instance = TRANSFORM_REGISTRY.create(name=name)
                    assert instance is not None, f'expected Callable, got None for transform {instance}'
                    result.append(instance)
            elif callable(transform):
                result.append(transform)
            else:
                raise MisconfigurationException(message=f'transform param {transform} is neither a dict or a callable')
    elif isinstance(transforms, dict):
        for key, value in transforms.items():
            if value is not None:
                instance = TRANSFORM_REGISTRY.create(name=key, **value)
                assert instance is not None, f'expected Callable, got None for transform {instance}'
                result.append(instance)
            else:
                instance = TRANSFORM_REGISTRY.create(name=key)
                assert instance is not None, f'expected Callable, got None for transform {instance}'
                result.append(instance)
    else:
        raise MisconfigurationException(message=f'transforms params {transforms} is not a list or dict but'
                                                f'has type {str(type(transforms))}')
    return result
