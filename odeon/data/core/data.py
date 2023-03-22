"""
"""
# from odeon import LOGGER
# from odeon.core.exceptions import MisconfigurationException
from odeon.core.registry import GenericRegistry
from odeon.core.types import OdnData


@GenericRegistry.register('data', aliases=['data_module'])
class DataRegistry(GenericRegistry[OdnData]):
    @classmethod
    def register_fn(cls, cl: OdnData, name: str):

        cls._registry[name] = cl
