"""odeon package
"""
import pathlib
from collections.abc import Mapping
# from dataclasses introspection.py fields
from pathlib import Path
from typing import Any, List, Optional

from jsonargparse import set_config_read_mode
from jsonargparse import ArgumentParser
# from omegaconf introspection.py OmegaConf

from odeon.core.app import APP_REGISTRY
from odeon.core.registry import GenericRegistry
from odeon.data import (ALBU_TRANSFORM_REGISTRY, DATA_REGISTRY, Input)
from odeon.models import (MODEL_REGISTRY, ChangeUnet, SegmentationModule)

from .core.default_path import ODEON_ENV, ODEON_PATH
from .core.env import Env, EnvConf, get_env_variable
from .core.io_utils import create_empty_file, create_path_if_not_exists
from .core.types import PARSER
# from .fit introspection.py FitApp, fit_plugin, pl_callback_plugin, pl_logger_plugin
# TODO load plugins
from .metrics import METRIC_REGISTRY

__all__ = ['Env', 'ODEON_ENV', 'ODEON_PATH', 'MODEL_REGISTRY', 'METRIC_REGISTRY',
           'SegmentationModule', 'ChangeUnet', 'APP_REGISTRY',
           'GenericRegistry', 'DATA_REGISTRY', 'Input', 'ALBU_TRANSFORM_REGISTRY']

# DEFAULT_ODEON_PATH: Path = HOME
_this_dir: Path = pathlib.Path(__file__).resolve().parent
_URL_ENABLED: bool = True
set_config_read_mode(urls_enabled=_URL_ENABLED)  # enable URL as path file
_DEFAULT_PARSER: PARSER = 'yaml'
_ODEON_PARSE_ENV_VARIABLE = 'ODEON_PARSER'
_ODEON_PARSER_AVAILABLE: List[PARSER] = ['yaml', 'jsonnet', 'omegaconf']


def _parser_type() -> PARSER:
    v: Optional[Any] = get_env_variable(_ODEON_PARSE_ENV_VARIABLE)
    if v is not None and v in _ODEON_PARSER_AVAILABLE:
        return v
    else:
        return _DEFAULT_PARSER


def bootstrap() -> Env:
    """
    Used to check if the .odeon directory is created and env.yml is inside.
    Otherwise, it will create the necessary directory and config file associated

    Returns
    -------
    Env
    """

    create_path_if_not_exists(ODEON_PATH)
    parser = ArgumentParser()
    parser.add_dataclass_arguments(theclass=EnvConf, nested_key='--env')
    if ODEON_ENV.is_file():

        cfg = parser.parse_path(str(ODEON_ENV))
        instantiated_conf = parser.instantiate_classes(cfg=cfg)
        env = instantiated_conf.env
        """"
        schema = OmegaConf.structured(EnvConf)
        conf = OmegaConf.load(ODEON_ENV)
        conf = OmegaConf.merge(schema, conf)
        assert isinstance(conf, Mapping)
        env_conf: EnvConf = EnvConf(**conf)
        """

    else:
        # env_fields = fields(EnvConf())
        # end_d = {field.name: field.value}
        with open('default_config.yaml', 'w') as file:
            # Exporting as YAML
            file.write(parser.dump(cfg=parser.parse_args([])))
        env = Env()

    return env


ENV = bootstrap()  # set the environment of application
with (_this_dir / ".." / "VERSION").open() as vf:
    __version__ = vf.read().strip()
