from typing import Dict, List, Type, Union, cast

from pytorch_lightning.callbacks.callback import Callback

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import OdnCallback


@GenericRegistry.register('callbacks', aliases=['calls', 'callback'])
class CallbackRegistry(GenericRegistry[Type[OdnCallback]]):
    @classmethod
    def register_fn(cls, cl: Type[OdnCallback], name: str):
        # assert isinstance(cl, Callback)
        cls._registry[name] = cl


def build_callbacks(callbacks: List[Union[Dict, Callback]]) -> List[Callback]:
    result: List[Callback] = list()
    for callback in callbacks:
        if isinstance(callback, dict):
            name = callback['name']
            if 'params' in callback:
                params: Dict = callback['params']
                result.append(cast(CallbackRegistry.create(name=name, **params), OdnCallback))
            else:
                result.append(cast(CallbackRegistry.create(name=name), OdnCallback))
        elif callable(Callback):
            result.append(callback)
        else:
            raise MisconfigurationException(message=f'callback param {callback} is neither a dict or a  PL callback')
    return result
