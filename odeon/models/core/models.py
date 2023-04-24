from typing import Dict, List, Union

from pytorch_lightning import LightningModule

from odeon.core.exceptions import MisconfigurationException
from odeon.core.logger import get_logger
from odeon.core.registry import GenericRegistry

LOGGER = get_logger(logger_name=__name__)


MODEL_REGISTRY = GenericRegistry[LightningModule]


def build_models(models: List[Union[Dict, LightningModule]]) -> List[LightningModule]:
    result: List[LightningModule] = list()
    for model in models:
        if isinstance(model, dict):
            name = model['name']
            if 'params' in model:
                params: Dict = model['params']
                instance = MODEL_REGISTRY.create(name=name, **params)
                assert instance is not None
                result.append(instance)
            else:
                instance = MODEL_REGISTRY.create(name=name)
                assert instance is not None
                result.append(instance)
        elif callable(LightningModule):
            result.append(model)
        else:
            raise MisconfigurationException(message=f'model param {model} is neither a dict or a  PL module')
    return result
