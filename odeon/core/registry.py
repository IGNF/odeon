import logging
from typing import Dict, Generic, List, Optional, Protocol, TypeVar, Union

T = TypeVar("T")
V = TypeVar("V")
logger = logging.getLogger()


class RegistryMixin(Protocol):
    """ Registry base class"""

    _registry: Dict[str, T] = {}

    @classmethod
    def register(cls,
                 name: str,
                 aliases: Optional[Union[str, List[str]]] = None
                 ) -> T:
        """

        Parameters
        ----------
        name: str
        aliases: Optional[Union[str, List[str]]]
        Returns
        -------
        value: T (type variable)
        """
        ...

    @classmethod
    def get(cls, name: str) -> T:
        """
        """
        ...

    @classmethod
    def get_registry(cls):
        ...


class FactoryMixin(Protocol):
    """ Behavioural Protocol for """
    @classmethod
    def create(cls, name: str, **kwargs):
        ...


class GenericRegistry(RegistryMixin, FactoryMixin, Generic[T]):

    @classmethod
    def get(cls, name: str):
        """
        """
        return cls._registry[name]

    @classmethod
    def register(cls, name: str, aliases: Optional[Union[str, List[str]]] = None) -> T:
        def inner_wrapper(wrapped_class: T) -> T:
            if name in cls._registry:
                logger.warning('Executor %s already exists. Will replace it', name)
            cls.register_class(cl=wrapped_class, name=name, aliases=aliases)
            return wrapped_class
        return inner_wrapper

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
    def get_registry(cls) -> T:
        return cls._registry

    @classmethod
    def create(cls, name: str, **kwargs) -> T:
        """ Factory command to create the executor.
        This method gets the appropriate Executor class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            name (str): The name of the executor to create.
        Returns:
            An instance of the executor that is created.
        """

        if name not in cls._registry:
            logger.warning('Executor %s does not exist in the registry', name)
            return None

        _class = cls._registry[name]
        _instance = _class(**kwargs)
        return _instance
