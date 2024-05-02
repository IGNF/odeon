from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
# from .rio import RioEngine
from layers.core.types import BOUNDS

from odeon.core.types import PARAMS, URI

from .raster import Raster
from .rio import RioEngine


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class Mask(Raster):
    one_hot_encoded: bool = False
    _is_one_hot_encoded: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.engine == 'rio':
            self._engine = RioEngine(**self.engine_params)
        else:
            raise ValueError(f'engine {self.engine} not supported')
        self._is_one_hot_encoded = self.one_hot_encoded

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
              path: Optional[URI] = None,
              bounds: Optional[BOUNDS] = None,
              options: Optional[PARAMS] = None,):
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
