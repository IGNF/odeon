from typing import Any, Dict, Optional, Protocol, runtime_checkable

from odeon.core.types import PARAMS


@runtime_checkable
class GeoIO(Protocol):

    def is_geo_referenced(self) -> bool:
        ...

    def read(self,
             *args,
             **kwargs,) -> Any:
        ...

    def write(self, data: Any, *args, **kwargs):
        ...


@runtime_checkable
class Modality(Protocol):

    def is_geo_referenced(self) -> bool:
        ...

    def read(self,
             *args,
             **kwargs, ) -> Any:
        ...

    def write(self, *args, **kwargs):
        ...


@runtime_checkable
class NormalizableModality(Protocol):

    def norm(self, *args, **kwargs) -> Any:
        ...

    def can_denormalize(self) -> bool:
        ...

    def denorm(self, *args, **kwargs) -> Any:
        ...


@runtime_checkable
class TransformableModality(NormalizableModality, Protocol):

    def transform(self) -> Any:
        ...


@runtime_checkable
class AlbumentationModality(TransformableModality, Protocol):

    def is_albu_transform(self) -> bool:
        ...

    def has_albu_pipeline(self) -> bool:
        ...


class Sample:
    """Sample class to represent a single sample in a dataset"""
    def __init__(self,
                 data: Optional[Dict[str, PARAMS]] = None):
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: PARAMS):
        self._data = value
