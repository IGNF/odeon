from typing import Dict, List, Union, cast

from pytorch_lightning.callbacks.callback import Callback

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import PARAMS, OdnCallback

CALLBACK_REGISTRY = GenericRegistry[OdnCallback]


def build_callbacks(callbacks: List[Union[PARAMS, Callback]] | Dict[str, PARAMS]) -> List[Callback]:
    result: List[Callback] = list()
    if isinstance(callbacks, list):
        for callback in callbacks:
            if isinstance(callback, dict):
                name = callback['name']
                if 'params' in callback:
                    params: Dict = callback['params']
                    result.append(cast(OdnCallback, CALLBACK_REGISTRY .create(name=name, **params)))
                else:
                    result.append(cast(OdnCallback, CALLBACK_REGISTRY .create(name=name)))
            elif callable(Callback):
                result.append(callback)
            else:
                raise MisconfigurationException(message=f'callback '
                                                        f'param {callback} is neither a dict or a  PL callback')
    elif isinstance(callbacks, dict):
        for key, value in callbacks.items():
            if value:
                result.append(cast(OdnCallback, CALLBACK_REGISTRY .create(name=key, **value)))
            else:
                result.append(cast(OdnCallback, CALLBACK_REGISTRY.create(name=key)))
    else:
        raise MisconfigurationException(message=f'expected callbacks {callbacks} params to be either a dict or a list')
    return result
