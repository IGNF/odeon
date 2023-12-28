from typing import Dict, List, Union

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS
from odeon.core.logger import get_logger
from odeon.core.python_env import debug_mode

from .core.types import OdnCallback


logger = get_logger(__name__, debug=debug_mode)


class CallbackRegistry(GenericRegistry[type[OdnCallback]]):
    _registry: Dict[str, type[OdnCallback]] = {}
    _alias_registry: Dict[str, str] = {}

    @classmethod
    def create(cls, name: str, **kwargs) -> OdnCallback:
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


CALLBACK_REGISTRY = CallbackRegistry


def build_callbacks(callbacks: List[Union[PARAMS, OdnCallback]] | Dict[str, PARAMS]) -> List[OdnCallback]:
    result: List[OdnCallback] = list()
    if isinstance(callbacks, list):
        for callback in callbacks:
            if isinstance(callback, dict):
                name = callback['name']
                if 'params' in callback:
                    params: Dict = callback['params']
                    result.append(CALLBACK_REGISTRY.create(name=name, **params))
                else:
                    result.append(CALLBACK_REGISTRY .create(name=name))
            elif isinstance(callback, OdnCallback):
                result.append(callback)
            else:
                raise MisconfigurationException(message=f'callback '
                                                        f'param {callback} is neither a dict or a  PL callback')
    elif isinstance(callbacks, dict):
        for key, value in callbacks.items():
            if value:
                result.append(CALLBACK_REGISTRY.create(name=key, **value))
            else:
                result.append(CALLBACK_REGISTRY.create(name=key))
    else:
        raise MisconfigurationException(message=f'expected callbacks {callbacks} params to be either a dict or a list')
    return result
