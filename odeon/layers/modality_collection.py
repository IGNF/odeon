from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from odeon.core.logger import get_logger
from odeon.core.types import PARAMS

from ._modality import Modality
from .registry import ModalityRegistry
from .types import BOUNDS

logger = get_logger(__name__)


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class ModalityCollection:

    modality_params: Dict[str, PARAMS | Modality]
    geo_referenced: bool = False
    _modalities: Dict[str, Modality] = field(init=False)
    _data: PARAMS = field(init=False)

    def __post_init__(self):
        for k, v in self.modality_params.items():
            if isinstance(v, Modality):
                self._modalities[k] = v
            else:
                self._modalities[k] = ModalityRegistry.create(name=k, **v)

    def is_geo_referenced(self) -> bool:
        return self.geo_referenced

    @property
    def modalities(self) -> Dict[str, Modality]:
        return self._modalities

    @modalities.setter
    def modalities(self,
                   value: Dict[str, Modality],):
        self._modalities = value

    @property
    def data(self) -> PARAMS:
        return self._data

    @data.setter
    def data(self,
             value: PARAMS,):
        self._data = value

    def read(self,
             bounds: Optional[BOUNDS] = None,
             *args,
             **kwargs,) -> PARAMS:
        for k, v in self.modalities.items():
            if self.is_geo_referenced():
                self.data[k] = v.read(bounds=bounds, *args, **kwargs)
            else:
                self.data[k] = v.read(*args, **kwargs)
        return self.data

    def write(self, data: Dict[str, Any], bounds: Optional[BOUNDS] = None, *args, **kwargs):
        for k, v in data.items():
            assert k in self.modalities
            if self.is_geo_referenced() and self.modalities[k].is_geo_referenced():
                self.modalities[k].write(data=v, bounds=bounds, *args, **kwargs)
            else:
                self.modalities[k].write(data=v, *args, **kwargs)
