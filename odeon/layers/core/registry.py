from typing import Dict, Type

from layers.core.modality import Modality

from odeon.core.introspection import instanciate_class
from odeon.core.logger import get_logger
from odeon.core.registry import GenericRegistry

logger = get_logger(__name__)


class ModalityRegistry(GenericRegistry[Type[Modality]]):
    _registry: Dict[str, Type[Modality]] = {}

    @classmethod
    def create(cls, name: str, **kwargs) -> Modality:
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
         Modality: An instance of the executor that is created.
        """

        if name not in cls._registry:
            logger.error(f"{name} not registered in registry {str(name)}")
            raise KeyError()
        _class = cls.get(name=name)
        _instance = instanciate_class(_class, **kwargs)
        assert isinstance(_instance, Modality)
        return _instance


MODALITY_REGISTRY = ModalityRegistry
