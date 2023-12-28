from typing import Dict, List, Union

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS
from odeon.core.logger import get_logger
from odeon.core.python_env import debug_mode

from .core.types import OdnLogger
logger = get_logger(__name__, debug=debug_mode)


class LoggerRegistry(GenericRegistry[type[OdnLogger]]):
    _registry: Dict[str, type[OdnLogger]] = {}
    _alias_registry: Dict[str, str] = {}

    @classmethod
    def create(cls, name: str, **kwargs) -> OdnLogger:
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
         OdnLogger: An instance of the executor that is created.
        """

        if name not in cls._registry:
            logger.error(f"{name} not registered in registry {str(cls.__name__)}")
            raise KeyError()
        _class = cls.get(name=name)
        return _class(**kwargs)


LOGGER_REGISTRY = LoggerRegistry


def build_loggers(
        loggers: List[Union[Dict, OdnLogger]] | Dict[str, PARAMS] | OdnLogger | bool
) -> List[OdnLogger] | bool | OdnLogger:
    result: List[OdnLogger] = list()
    if isinstance(loggers, list):
        for logger in loggers:
            if isinstance(logger, dict):
                name = logger['name']
                if 'params' in logger:
                    params: Dict = logger['params']
                    result.append(LOGGER_REGISTRY.create(name=name, **params))
                else:
                    result.append(LOGGER_REGISTRY.create(name=name))
            elif callable(OdnLogger):
                result.append(logger)
            else:
                raise MisconfigurationException(message=f'logger param {logger} is neither a dict or a  PL Logger')
    elif isinstance(loggers, dict):
        for key, value in loggers.items():
            if value is not None:
                result.append(LOGGER_REGISTRY.create(name=key, **value))
            else:
                result.append(LOGGER_REGISTRY.create(name=key))
    elif isinstance(loggers, OdnLogger):
        return loggers
    elif isinstance(loggers, bool):
        return loggers
    else:
        raise MisconfigurationException(message=f'loggers params {loggers} is neither a list or a dict')
    return result
