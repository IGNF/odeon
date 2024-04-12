from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from odeon.core.types import URI

from ._engine import Engine
from .data import DTYPE_MAX, InputDType
# from .rio import RioEngine
from .types import BOUNDS

# import rasterio as rio


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class Raster:
    band_indices = None
    boundless: bool = True
    height: Optional[int] = None
    width: Optional[int] = None
    dtype_max: float = DTYPE_MAX[InputDType.UINT8.value]
    mean: Optional[List] = field(default=None)
    std: Optional[List] = field(default=None)
    # resampling: rio.enums.Resampling = rio.enums.Resampling.nearest
    _engine: Engine = field(init=False)
    _is_albu_transform: bool = field(init=False, default=True)
    _is_geo_referenced: bool = field(init=False, default=True)

    def is_geo_referenced(self) -> bool:
        return self._is_geo_referenced

    def read(self,
             path: URI,
             bounds: Optional[BOUNDS] = None,
             *args,
             **kwargs, ) -> np.ndarray:
        raise NotImplementedError('Not implemented yet')

    def write(self, data: np.ndarray, bounds: Optional[BOUNDS], *args, **kwargs):
        raise NotImplementedError('Not implemented yet')

    def is_albu_transform(self) -> bool:
        return self._is_albu_transform

    def norm(self, *args, **kwargs) -> Any:
        raise NotImplementedError('Not implemented yet')

    def can_denormalize(self) -> bool:
        raise NotImplementedError('Not implemented yet')

    def denorm(self, *args, **kwargs) -> Any:
        raise NotImplementedError('Not implemented yet')
