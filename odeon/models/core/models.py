from typing import Dict, List, Union

from pytorch_lightning import LightningModule

from odeon.core.exceptions import MisconfigurationException
from odeon.core.logger import get_logger
from odeon.core.registry import GenericRegistry

LOGGER = get_logger(logger_name=__name__)


class ModelRegistry(GenericRegistry[LightningModule]):

    _registry: Dict[str, LightningModule] = {}
    __name__ = 'model registry'

    @classmethod
    def create(cls, name: str, **kwargs) -> LightningModule:
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
         App: An instance of the executor that is created.
        """

        if name not in cls._registry:
            LOGGER.error(f"{name} not registered in registry {str(name)}")
            raise KeyError()

        _class = cls.get(name=name)
        _instance = _class(**kwargs)
        return _instance


MODEL_REGISTRY = ModelRegistry


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
