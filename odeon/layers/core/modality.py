from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from odeon.core.abstract_interface import AbstractDataclass
from odeon.core.types import PARAMS


@runtime_checkable
class GeoIO(Protocol):

    @abstractmethod
    def is_geo_referenced(self) -> bool:
        ...

    @abstractmethod
    def read(self,
             *args,
             **kwargs,) -> Any:
        ...

    @abstractmethod
    def write(self, data: Any, *args, **kwargs):
        ...

    @staticmethod
    def get_name():
        raise NotImplementedError('the get_name function has to be implemented for each new modality class')


@runtime_checkable
class Modality(Protocol):

    @abstractmethod
    def is_geo_referenced(self) -> bool:
        ...

    @abstractmethod
    def read(self,
             *args,
             **kwargs, ) -> Any:
        ...

    @abstractmethod
    def write(self, *args, **kwargs,):
        ...


@runtime_checkable
class NormalizableModality(Protocol):

    @abstractmethod
    def norm(self, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def can_denormalize(self) -> bool:
        ...

    @abstractmethod
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


@dataclass
class ModalityFactory:
    name: str
    dtype: str = 'uint8'


@dataclass
class ModalityCollectionFactory:
    modalities: Dict[str, ModalityFactory]


@dataclass
class AbsModality(AbstractDataclass):
    name: str
    field_name: str


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
