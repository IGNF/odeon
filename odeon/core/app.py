from typing import Callable, Dict, List, Optional, Tuple

from jsonargparse import ArgumentParser, Namespace

from .registry import GenericRegistry
from .singleton import Singleton


class App(metaclass=Singleton):
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


class AppRegistry(GenericRegistry[App]):
    _registry: Dict[str, App] = {}


APP_REGISTRY = AppRegistry
GenericRegistry.register_class(cl=APP_REGISTRY, name='app_registry', aliases=['apps', 'odn_app', 'app'])
