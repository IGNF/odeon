from typing import (Any, Dict, Generic, List, Optional, Protocol, TypeVar,
                    Union, cast)

from odeon.core.logger import get_logger

T = TypeVar("T", bound=Any)
V = TypeVar("V", bound=Any)
logger = get_logger(__name__)


class FactoryMixin(Protocol):
    """ Behavioural Protocol for """
    @classmethod
    def create(cls, name: str, **kwargs):
        ...


class GenericRegistry(FactoryMixin, Generic[T]):
    _registry: Dict[str, T] = {}

    @classmethod
    def get(cls, name: str) -> T:
        """

        Parameters
        ----------
        name

        Returns
        -------

        """
        return cls._registry[name]

    @classmethod
    def register(cls, name: str, aliases: Optional[Union[str, List[str]]] = None) -> T:
        def inner_wrapper(wrapped_class: T) -> T:
            if name in cls._registry:
                logger.warning(
                    f'{wrapped_class} has one name or alias ({name}) in already existing in registry {cls} .'
                    f' It Will replace it, old class {cls._registry[name]}')
            cls.register_class(cl=wrapped_class, name=name, aliases=aliases)
            return wrapped_class
        return cast(T, inner_wrapper)

    @classmethod
    def register_class(cls, cl: T, name: str = 'none', aliases: Optional[Union[str, List[str]]] = None):
        if name != 'none':
            if name in cls._registry:
                raise KeyError(f'name {name} already in Registry {str(cls)}')
            else:
                cls.register_fn(cl=cl, name=name)
        if isinstance(aliases, str):
            if aliases in cls._registry:
                raise KeyError(f'alias {aliases} already in Registry {str(cls)}')
            else:
                cls.register_fn(cl=cl, name=name)
        elif isinstance(aliases, List):
            if len(aliases) > 0:
                cls.register_class(cl, name='none', aliases=aliases[1:])
        else:
            pass

    @classmethod
    def register_fn(cls, cl: T, name: str):
        cls._registry[name] = cl

    @classmethod
    def get_registry(cls) -> Dict[str, T]:
        return cls._registry

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
