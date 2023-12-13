from typing import (Any, Dict, Generic, List, Optional, TypeVar,
                    Union, cast)

from odeon.core.logger import get_logger

T = TypeVar("T", bound=Any)
V = TypeVar("V", bound=Any)
logger = get_logger(__name__)


class MetaRegistry(type):

    def __str__(self):
        return self.__class__.__name__


class GenericRegistry(Generic[T], metaclass=MetaRegistry):
    _registry: Dict[str, T] = {}
    _alias_registry: Dict[str, str] = {}
    __name__ = f"registry of type {str(T)}"

    @classmethod
    def get_registry(cls) -> Dict[str, T]:
        return cls._registry

    @classmethod
    def get_aliases_registry(cls) -> Dict[str, T]:
        return cls._alias_registry

    @classmethod
    def get(cls, name: str) -> T:
        """

        Parameters
        ----------
        name

        Returns
        -------

        Raises
        ------
        KeyError if name is not registered
        """
        if name in cls._registry.keys():
            return cls._registry[name]
        elif name in cls._alias_registry.keys():
            return cls._registry[cls._alias_registry[name]]
        else:
            raise KeyError(f'Unknown name for registry{str(cls)}')

    @classmethod
    def register(cls, name: str, aliases: Optional[Union[str, List[str]]] = None) -> T:
        def inner_wrapper(wrapped_class: T) -> T:

            cls.register_class(cl=wrapped_class, name=name)
            cls.register_aliases(name=name, aliases=aliases)
            return wrapped_class
        return cast(T, inner_wrapper)

    @classmethod
    def register_class(cls, cl: T, name: str = 'none'):
        if name != 'none':
            if name in cls._registry:
                raise KeyError(f'name {name} already in Registry {str(cls)}')
            else:
                cls.register_fn(cl=cl, name=name)
        else:
            pass

    @classmethod
    def register_aliases(cls, name: str, aliases: Optional[Union[str, List[str]]] = None):
        if name != 'none':
            if aliases is None:
                pass
            elif isinstance(aliases, str):
                cls.register_aliases_fn(alias=aliases, name=name)
            elif isinstance(aliases, list):
                for alias in aliases:
                    cls.register_aliases_fn(alias=str(alias), name=name)
        else:
            pass

    @classmethod
    def register_fn(cls, cl: T, name: str):
        cls._registry[name] = cl

    @classmethod
    def register_aliases_fn(cls, alias: str, name: str):
        cls._alias_registry[alias] = name

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[T]:
        """
        Factory command to create an instance.
        This method gets the appropriate Registered class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.

        Parameters
        ----------
         name: str, The name of the executor to create.
         kwargs

        Returns
        -------
         An instance of the executor that is created.
        """

        if name not in cls._registry:
            logger.warning(f'class {name} from registry {cls} does not exist in the registry')
            return None

        _class = cls._registry[name]
        _instance = _class(**kwargs)
        return _instance
