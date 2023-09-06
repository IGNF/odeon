from typing import Dict

from odeon.core.registry import GenericRegistry

from .types import OdnData


class DataRegistry(GenericRegistry[OdnData]):
    _registry: Dict[str, OdnData] = {}


DATA_REGISTRY = DataRegistry
GenericRegistry.register_class(cl=DATA_REGISTRY, name='data_registry', aliases=['data_module', 'data_module_registry'])
