from typing import Dict, List, Type, Union, cast

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS, OdnLogger


@GenericRegistry.register('logger', aliases=['log'])
class LoggerRegistry(GenericRegistry[Type[OdnLogger]]):
    @classmethod
    def register_fn(cls, cl: Type[OdnLogger], name: str):

        cls._registry[name] = cl


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
                    result.append(cast(LoggerRegistry.create(name=name, **params), OdnLogger))
                else:
                    result.append(cast(LoggerRegistry.create(name=name), OdnLogger))
            elif callable(OdnLogger):
                result.append(logger)
            else:
                raise MisconfigurationException(message=f'logger param {logger} is neither a dict or a  PL Logger')
    elif isinstance(loggers, dict):
        for key, value in loggers.items():
            if value is not None:
                result.append(cast(LoggerRegistry.create(name=key, **value), OdnLogger))
            else:
                result.append(cast(LoggerRegistry.create(name=key), OdnLogger))
    elif isinstance(loggers, OdnLogger):
        return loggers
    elif isinstance(loggers, bool):
        return loggers
    else:
        raise MisconfigurationException(message=f'loggers params {loggers} is neither a list or a dict')
    return result
