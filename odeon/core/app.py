from typing import Callable, Dict, List, Optional, Tuple

from jsonargparse import ArgumentParser, Namespace

from .logger import get_logger
from .python_env import debug_mode
from .registry import GenericRegistry

logger = get_logger(__name__, debug=debug_mode)


class App:
    """ abstract base class for any Odeon App like fit, feature, etc.
    Odeon apps are Singleton taking a dataclass configuration as argument """

    def run(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        ...

    @staticmethod
    def get_instance(parser: ArgumentParser, cfg: Namespace) -> Callable:
        instantiated_conf = parser.instantiate_classes(cfg=cfg)
        instance = instantiated_conf.conf
        return instance

    @staticmethod
    def parse_args(class_config: type, *args: List,
                   parser_mode: str) -> Tuple[ArgumentParser, Namespace, Optional[bool]]:
        parser: ArgumentParser = ArgumentParser(parser_mode=parser_mode)
        parser.add_dataclass_arguments(theclass=class_config, nested_key='conf', help=class_config.__doc__)
        parser.add_argument('--debug', type=bool, default=False)
        cfg = parser.parse_args(*args)
        DEBUG_MODE = cfg.debug
        return parser, cfg, DEBUG_MODE

    @staticmethod
    def get_class_config() -> type:
        ...


class AppRegistry(GenericRegistry[type[App]]):
    _registry: Dict[str, type[App]] = {}

    @classmethod
    def create(cls, name: str, **kwargs) -> App:
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
         App: An instance of the executor that is created.
        """

        if name not in cls._registry:
            logger.error(f"{name} not registered in registry {str(name)}")
            raise KeyError()

        _class = cls.get(name=name)
        _instance = _class(**kwargs)
        return _instance


APP_REGISTRY = AppRegistry
