from typing import Callable, Dict, List, Optional, Union, Any, Type

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS
from odeon.core.logger import get_logger
from odeon.core.python_env import debug_mode

logger = get_logger(__name__, debug=debug_mode)


class AlbuTransformRegistry(GenericRegistry[Type[Callable[..., Any]]]):
    _registry: Dict[str, Type[Callable[..., Any]]] = {}
    _alias_registry: Dict[str, str] = {}

    @classmethod
    def create(cls, name: str, **kwargs) -> Callable[..., Any]:
        """
        Factory command to create an instance.
        This method gets the appropriate Registered class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.

        Parameters
        ----------
         name: str, The name of the executor to create.
         kwargs

        Returns
        -------
         Callable: An instance of the executor that is created.
        """

        if name not in cls._registry:
            logger.error(f"{name} not registered in registry {str(name)}")
            raise KeyError()

        _class = cls.get(name=name)
        return _class(**kwargs)


ALBU_TRANSFORM_REGISTRY = AlbuTransformRegistry
# GenericRegistry.register_class(cl=ALBU_TRANSFORM_REGISTRY, name='albu_tranform_registry', aliases=['transform'])
ONE_OFF_NAME = 'one_off'
ONE_OFF_ALIASES = ['OneOff']


def build_transform(transforms: List[Union[Dict, Callable]] | Dict[str, PARAMS],
                    buffer: Optional[List[Callable]] = None) -> List[Callable]:
    result: List[Callable] = buffer if isinstance(buffer, list) else list()
    if isinstance(transforms, list):
        for transform in transforms:
            if isinstance(transform, dict):
                name = transform['name']
                if 'params' in transform:
                    params: Dict = transform['params']
                    instance = ALBU_TRANSFORM_REGISTRY.create(name=name, **params)
                    assert instance is not None, f'expected Callable, got None for transform {instance}'
                    result.append(instance)
                else:
                    instance = ALBU_TRANSFORM_REGISTRY.create(name=name)
                    assert instance is not None, f'expected Callable, got None for transform {instance}'
                    result.append(instance)
            elif callable(transform):
                result.append(transform)
            else:
                raise MisconfigurationException(message=f'transform param {transform} is neither a dict or a callable')
    elif isinstance(transforms, dict):
        for key, value in transforms.items():
            if value is not None:
                instance = ALBU_TRANSFORM_REGISTRY.create(name=key, **value)
                assert instance is not None, f'expected Callable, got None for transform {instance}'
                result.append(instance)
            else:
                instance = ALBU_TRANSFORM_REGISTRY.create(name=key)
                assert instance is not None, f'expected Callable, got None for transform {instance}'
                result.append(instance)
    else:
        raise MisconfigurationException(message=f'transforms params {transforms} is not a list or dict but'
                                                f'has type {str(type(transforms))}')
    return result
