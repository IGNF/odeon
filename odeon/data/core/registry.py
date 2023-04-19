from odeon.core.registry import GenericRegistry

from .types import OdnData


@GenericRegistry.register('data', aliases=['data_module'])
class DataRegistry(GenericRegistry[OdnData]):
    ...
