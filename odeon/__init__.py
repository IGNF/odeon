"""odeon package
"""
import pathlib
from collections.abc import Mapping
from logging import Logger
from pathlib import Path
from typing import Any, List, Optional, Tuple

import albumentations as A
from jsonargparse import set_config_read_mode
from omegaconf import OmegaConf

from .core.default_path import ODEON_ENV, ODEON_PATH
from .core.env import Env, EnvConf, get_env_variable
from .core.io_utils import create_empty_file, create_path_if_not_exists
from .core.logger import get_logger
from .core.types import PARSER
from .data.core import ONE_OFF_ALIASES, ONE_OFF_NAME, TRANSFORM_REGISTRY

# from .metrics import *

# from odeon.models import *


__all__ = ['Env', 'ODEON_ENV', 'ODEON_PATH']
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


def bootstrap() -> Tuple[Env, Logger]:
    """
    Used to check if the .odeon directory is created and env.yml is inside.
    Otherwise, it will create the necessary directory and config file associated

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
        create_empty_file(path=ODEON_ENV)
        env = Env(config=EnvConf())

    return env, get_logger(logger_name='odeon', debug=env.config.debug_mode)


ENV, LOGGER = bootstrap()  # set the environment of application
with (_this_dir / ".." / "VERSION").open() as vf:
    __version__ = vf.read().strip()

"""
TRANSFORM REGISTRY
"""

TRANSFORM_REGISTRY.register_class(A.VerticalFlip, name='vertical_flip', aliases=['v_flip'])
TRANSFORM_REGISTRY.register_class(A.HorizontalFlip, name='horizontal_flip', aliases=['h_flip'])
TRANSFORM_REGISTRY.register_class(A.Transpose, name='transpose')
TRANSFORM_REGISTRY.register_class(A.Rotate, name='rotate', aliases=['rot'])
TRANSFORM_REGISTRY.register_class(A.RandomRotate90, name='random_rotate_90', aliases=['rrotate90', 'rrot90'])
TRANSFORM_REGISTRY.register_class(A.resize, name='resize')
TRANSFORM_REGISTRY.register_class(A.RandomCrop, name='random_crop', aliases=['r_crop'])
TRANSFORM_REGISTRY.register_class(A.RandomResizedCrop, name='random_resize_crop', aliases=['rr_crop'])
TRANSFORM_REGISTRY.register_class(A.RandomSizedCrop, name='random_sized_crop', aliases=['rs_crop'])
TRANSFORM_REGISTRY.register_class(A.gaussian_blur, name='gaussian_blur', aliases=['g_blur'])
TRANSFORM_REGISTRY.register_class(A.gauss_noise, name='gauss_noise', aliases=['g_noise', 'gaussian_noise'])
TRANSFORM_REGISTRY.register_class(A.ColorJitter, name='color_jitter', aliases=['c_jitter', 'c_jit'])
TRANSFORM_REGISTRY.register_class(A.RandomGamma, name='random_gamma', aliases=['r_gamma'])
TRANSFORM_REGISTRY.register_class(A.OneOf, name=ONE_OFF_NAME, aliases=ONE_OFF_ALIASES)
