from typing import Dict, List, Type, Union, cast

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import OdnLogger


@GenericRegistry.register('logger', aliases=['log'])
class LoggerRegistry(GenericRegistry[Type[OdnLogger]]):
    @classmethod
    def register_fn(cls, cl: Type[OdnLogger], name: str):

        cls._registry[name] = cl


def build_loggers(loggers: List[Union[Dict, OdnLogger]]) -> List[OdnLogger]:
    result: List[OdnLogger] = list()
    for logger in loggers:
        if isinstance(logger, dict):
            name = logger['name']
            if 'params' in logger:
                params: Dict = logger['params']
                result.append(cast(LoggerRegistry.create(name=name, **params), OdnLogger))
            else:
                result.append(cast(LoggerRegistry.create(name=name), OdnLogger))
        elif callable(OdnLogger):
            result.append(logger)
        else:
            raise MisconfigurationException(message=f'logger param {logger} is neither a dict or a  PL Logger')
    return result
