"""main module

This module is used as command line entrypoint of odeon
The main function is the entrypoint.

this command line tools brings the ability to use an Odeon App registered in the app registry
from the command line. The app is configured with a configuration file (omegaconf by default,
the parameter of the parser mode is controlled with the Env class and the env configuration file)

odeon --help or -h gives access to the main help, which presents
the how-to of the cli, and gives the list of the available Apps

"""
from sys import argv
from typing import Callable, List, Optional, Tuple

from jsonargparse import ArgumentParser, Namespace

from odeon import ENV
from odeon.core.app import APP_REGISTRY


def main():

    app_name: str = argv[1]
    app_args: List = argv[2:]
    app = APP_REGISTRY.get(app_name)
    parser, cfg, debug = parse_args(app, app_args)
    # logger = get_logger(__name__, debug=debug)
    instance = get_instance(parser=ENV.config.config_parser, cfg=cfg)
    instance()


def parse_args(some_class, *args) -> Tuple[ArgumentParser, Namespace, Optional[bool]]:

    parser: ArgumentParser = ArgumentParser(parser_mode='omegaconf')
    parser.add_dataclass_arguments(theclass=some_class, nested_key='conf')
    parser.add_argument('--debug', type=bool, default=False)
    cfg = parser.parse_args(*args)
    DEBUG_MODE = cfg.debug
    return parser, cfg, DEBUG_MODE


def get_instance(parser: ArgumentParser, cfg: Namespace, *args, **kwargs) -> Callable:
    instantiated_conf = parser.instantiate_classes(cfg=cfg)
    instance = instantiated_conf.conf
    return instance


if __name__ == '__main__':

    main()
