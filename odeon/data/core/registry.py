from odeon.core.registry import GenericRegistry

from .types import OdnData

DATA_REGISTRY = GenericRegistry[OdnData]
GenericRegistry.register_class(cl=DATA_REGISTRY, name='data_registry', aliases=['data_module', 'data_module_registry'])
