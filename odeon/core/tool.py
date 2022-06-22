"""

"""
from dataclasses import dataclass
from .exception import ODNError
from abc import ABCMeta, abstractmethod


@dataclass
class ODNTool(metaclass=ABCMeta):
    @abstractmethod
    def run(self, *args, **kwargs) -> ODNError:
        ...

    def __call__(self, *args, **kwargs) -> ODNError:
        return self.run(args, kwargs)
