"""odeon package
"""
import pathlib
from collections.abc import Mapping
# from dataclasses import fields
from pathlib import Path
from typing import Any, List, Optional

from jsonargparse import set_config_read_mode
from omegaconf import OmegaConf

from odeon.core.app import APP_REGISTRY
from odeon.core.registry import GenericRegistry
from odeon.data import (ALBU_TRANSFORM_REGISTRY, DATA_REGISTRY, Input,
                        albu_transform_plugin, data_plugin)
from odeon.models import (MODEL_REGISTRY, ChangeUnet, SegmentationModule,
                          model_plugin)

from .core.default_path import ODEON_ENV, ODEON_PATH
from .core.env import Env, EnvConf, get_env_variable
from .core.io_utils import create_empty_file, create_path_if_not_exists
from .core.types import PARSER
from .fit import FitApp, fit_plugin, pl_callback_plugin, pl_logger_plugin
# TODO load plugins
from .metrics import (METRIC_REGISTRY, binary_metric_plugin,
                      multiclass_metric_plugin, multilabel_metric_plugin)

__all__ = ['Env', 'ODEON_ENV', 'ODEON_PATH', 'binary_metric_plugin', 'multilabel_metric_plugin',
           'multiclass_metric_plugin', 'model_plugin', 'MODEL_REGISTRY', 'METRIC_REGISTRY', 'SegmentationModule',
           'ChangeUnet', 'FitApp', 'fit_plugin', 'pl_callback_plugin', 'pl_logger_plugin', 'APP_REGISTRY',
           'GenericRegistry', 'DATA_REGISTRY', 'Input', 'data_plugin', 'ALBU_TRANSFORM_REGISTRY',
           'albu_transform_plugin']

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

    if ODEON_ENV.is_file():
        """
        parser = ArgumentParser()
        parser.add_argument('--env', type=EnvConf)
        conf = parser.parse_path(str(ODEON_ENV))
        env_conf: EnvConf = EnvConf(**dict(conf.env))
        """
        schema = OmegaConf.structured(EnvConf)
        conf = OmegaConf.load(ODEON_ENV)
        conf = OmegaConf.merge(schema, conf)
        assert isinstance(conf, Mapping)
        env_conf: EnvConf = EnvConf(**conf)
        env: Env = Env(config=env_conf)
    else:
        # env_fields = fields(EnvConf())
        # end_d = {field.name: field.value}
        create_empty_file(path=ODEON_ENV)
        env = Env()

    return env


ENV = bootstrap()  # set the environment of application
with (_this_dir / ".." / "VERSION").open() as vf:
    __version__ = vf.read().strip()
