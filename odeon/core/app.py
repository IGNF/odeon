from typing import Callable, List, Optional, Tuple

from jsonargparse import ArgumentParser, Namespace

from .registry import GenericRegistry
from .singleton import Singleton


class App(metaclass=Singleton):
    """ abstract base class for any Odeon App like fit, feature, etc.
    Odeon apps are Singleton"""

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
    def parse_args(class_config: type, *args: List) -> Tuple[ArgumentParser, Namespace, Optional[bool]]:
        parser: ArgumentParser = ArgumentParser(parser_mode='omegaconf')
        parser.add_dataclass_arguments(theclass=class_config, nested_key='conf')
        parser.add_argument('--debug', type=bool, default=False)
        cfg = parser.parse_args(*args)
        DEBUG_MODE = cfg.debug
        return parser, cfg, DEBUG_MODE

    def get_class_config(self) -> type:
        ...


APP_REGISTRY = GenericRegistry[App]
