from typing import Dict, List, Union, cast

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS, OdnLogger

LOGGER_REGISTRY = GenericRegistry[OdnLogger]


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
                    result.append(cast(OdnLogger, LOGGER_REGISTRY.create(name=name, **params)))
                else:
                    result.append(cast(OdnLogger, LOGGER_REGISTRY.create(name=name)))
            elif callable(OdnLogger):
                result.append(logger)
            else:
                raise MisconfigurationException(message=f'logger param {logger} is neither a dict or a  PL Logger')
    elif isinstance(loggers, dict):
        for key, value in loggers.items():
            if value is not None:
                result.append(cast(OdnLogger, LOGGER_REGISTRY.create(name=key, **value)))
            else:
                result.append(cast(OdnLogger, LOGGER_REGISTRY.create(name=key)))
    elif isinstance(loggers, OdnLogger):
        return loggers
    elif isinstance(loggers, bool):
        return loggers
    else:
        raise MisconfigurationException(message=f'loggers params {loggers} is neither a list or a dict')
    return result
