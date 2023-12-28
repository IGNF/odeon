"""odeon package
"""
import pathlib
from pathlib import Path

from .core.python_env import debug_mode
from .core.logger import get_logger
from .core.env import get_env, Env

from .core.default_path import ODEON_ENV, ODEON_PATH

logger = get_logger(__name__, debug=debug_mode)
ENV: Env = get_env()
ENV.load_plugins()  # load the configured plugins
logger.debug(f'env plugins: {ENV.plugins}')

__all__ = ['ENV', 'ODEON_ENV', 'ODEON_PATH']

# from .fit introspection.py FitApp, fit_plugin, pl_callback_plugin, pl_logger_plugin


# DEFAULT_ODEON_PATH: Path = HOME
_this_dir: Path = pathlib.Path(__file__).resolve().parent

with (_this_dir / ".." / "VERSION").open() as vf:
    __version__ = vf.read().strip()
