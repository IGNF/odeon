from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data import RASTER_ACCEPTED_EXTENSION
from .singleton import Singleton


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class KeyResolver(metaclass=Singleton):
    data_keys: Dict

    """
    def __post_init__(self):
        values = [value["name"] for value in self.data_keys.values()]
        keys = [key for key in self.data_keys.keys()]
        self._reverse_data_field = dict(zip(values, keys))
    """
    def get_value(self, input_key: str, name_field: Optional[str] = None) -> Any:
        value = self.data_keys[input_key] if name_field is None else self.data_keys[input_key][field]
        return value

    def get_data_kets(self) -> Dict:
        return self.data_keys

    def get_fields_for_type(self, data: str) -> List:

        return [{key: value} for key, value in self.data_keys.items() if value["type"] == data]


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class AppConfig:

    raster_accepted_prefix: List[str] = field(default_factory=lambda: RASTER_ACCEPTED_EXTENSION)

    def __post_init__(self):

        self.raster_accepted_prefix = list(set(self.raster_accepted_prefix + RASTER_ACCEPTED_EXTENSION))