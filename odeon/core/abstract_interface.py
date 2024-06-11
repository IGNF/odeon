from abc import ABC
from dataclasses import dataclass


@dataclass
class AbstractDataclass(ABC):
    """
    elegant way to define an Abstract base class that defines the interface for abstract dataclass
    allowing to avoid abstract class instantiation
    see post: https://stackoverflow.com/questions/60590442/
    abstract-dataclass-without-abstract-methods-in-python-prohibit-instantiation
    Custom version to avoid instantiation with any dataclass with a name starting with
    prefix Abs
    """
    def __new__(cls, *args, **kwargs):
        if str(cls.__name__).startswith('Abs') or cls.__bases__[0] == AbstractDataclass:
            raise TypeError(f"Cannot instantiate abstract dataclass {cls.__name__}. Any"
                            f"class starting with prefix Abs cannot be instantiated")
        return super().__new__(cls)
