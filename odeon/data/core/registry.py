from typing import Dict, Type

from odeon.core.registry import GenericRegistry
from odeon.core.logger import get_logger
from odeon.core.python_env import debug_mode

from .types import OdnData

logger = get_logger(__name__, debug=debug_mode)


class DataRegistry(GenericRegistry[Type[OdnData]]):
    _registry: Dict[str, Type[OdnData]] = {}

    @classmethod
    def create(cls, name: str, **kwargs) -> OdnData:
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
         OdnData: An instance of the executor that is created.
        """

        if name not in cls._registry:
            logger.error(f"{name} not registered in registry {str(name)}")
            raise KeyError()
        _class = cls.get(name=name)
        return _class(**kwargs)


DATA_REGISTRY = DataRegistry
