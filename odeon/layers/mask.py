from dataclasses import dataclass, field
from typing import Any

# from .rio import RioEngine
from .raster import RASTER_ENGINE, Raster


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class Mask(Raster):
    one_hot_encoded: bool = False
    _is_one_hot_encoded: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.engine in RASTER_ENGINE.keys():
            self._engine = RASTER_ENGINE[self.engine](**self.engine_params)
        else:
            raise ValueError(f'engine {self.engine} not supported,'
                             f' supported engines are {" - ".join(RASTER_ENGINE.keys())}')
        self._is_one_hot_encoded = self.one_hot_encoded

    def norm(self, *args, **kwargs) -> Any:
        raise NotImplementedError('Not implemented yet')

    def can_denormalize(self) -> bool:
        raise NotImplementedError('Not implemented yet')

    def denorm(self, *args, **kwargs) -> Any:
        raise NotImplementedError('Not implemented yet')
