from abc import ABC, abstractmethod
from typing import Any

# from odeon.core.types import PARAMS


class Engine(ABC):

    @abstractmethod
    def is_geo_referenced(self) -> bool:
        ...

    @abstractmethod
    def read(self,
             *args,
             **kwargs,) -> Any:
        ...

    @abstractmethod
    def write(self, *args, **kwargs):
        ...
