from typing import Dict, List, Union

from pytorch_lightning import LightningModule

from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry


@GenericRegistry.register('models', aliases=['pl_module', 'model', 'lightning_module'])
class ModelRegistry(GenericRegistry[LightningModule]):
    @classmethod
    def register_fn(cls, cl: LightningModule, name: str):
        print(type(cl))
        cls._registry[name] = cl


def build_callbacks(models: List[Union[Dict, LightningModule]]) -> List[LightningModule]:
    result: List[LightningModule] = list()
    for model in models:
        if isinstance(model, dict):
            name = model['name']
            if 'params' in model:
                params: Dict = model['params']
                result.append(ModelRegistry.create(name=name, **params))
            else:
                result.append(ModelRegistry.create(name=name))
        elif callable(LightningModule):
            result.append(model)
        else:
            raise MisconfigurationException(message=f'model param {model} is neither a dict or a  PL module')
    return result
