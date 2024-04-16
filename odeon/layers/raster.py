from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from odeon.core.types import PARAMS, URI

from ._engine import Engine
from .data import InputDType
from .rio import RioEngine
# from .rio import RioEngine
from .types import BOUNDS


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class Raster:
    engine: str = 'rio'
    engine_params: PARAMS = field(default_factory=lambda: dict())
    dtype: str = field(default=str(InputDType.UINT8.value))
    mean: Optional[List] = field(default=None)
    std: Optional[List] = field(default=None)
    # resampling: rio.enums.Resampling = rio.enums.Resampling.nearest
    _engine: Engine = field(init=False)
    _is_albu_transform: bool = field(init=False, default=True)
    _is_geo_referenced: bool = field(init=False, default=True)

    def __post_init__(self):
        if self.engine == 'rio':
            self._engine = RioEngine(**self.engine_params)
        else:
            raise ValueError(f'engine {self.engine} not supported')

    def is_geo_referenced(self) -> bool:
        return self._is_geo_referenced

    def read(self,
             path: Optional[URI] = None,
             bounds: Optional[BOUNDS] = None,
             options: Optional[PARAMS] = None) -> np.ndarray:
        if options:
            return self._engine.read(path, bounds, **options)
        else:
            return self._engine.read(path, bounds)

    def write(self,
              data: np.ndarray,
              bounds: Optional[BOUNDS] = None,
              options: Optional[PARAMS] = None):
        if options:
            self._engine.write(data, bounds, **options)
        else:
            self._engine.write(data, bounds)

    def is_albu_transform(self) -> bool:
        return self._is_albu_transform

    def norm(self, *args, **kwargs) -> Any:
        raise NotImplementedError('Not implemented yet')

    def can_denormalize(self) -> bool:
        raise NotImplementedError('Not implemented yet')

    def denorm(self, *args, **kwargs) -> Any:
        raise NotImplementedError('Not implemented yet')
