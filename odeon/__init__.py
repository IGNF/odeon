"""odeon package
"""
import pathlib
from pathlib import Path

from omegaconf import OmegaConf

from .core.python_env import debug_mode
from .core.logger import get_logger
from .core.env import get_env, Env
from .core.default_path import ODEON_ENV, ODEON_PATH

__name__ = 'odeon'

# DEFAULT_ODEON_PATH: Path = HOME
_this_dir: Path = pathlib.Path(__file__).resolve().parent
with (_this_dir / ".." / "VERSION").open() as vf:
    __version__ = vf.read().strip()


logger = get_logger(__name__, debug=debug_mode)
ENV: Env = get_env()
ENV.load_plugins()  # load the configured plugins
logger.debug(f'env plugins: {ENV.plugins}')


def store_resolver(store_name: str) -> str:
    match store_name:
        case 'user_config_store':
            return str(ENV.user_config_store)
        case 'user_artefact_store':
            return str(ENV.user_artefact_store)
        case 'user_model_store':
            return str(ENV.user_model_store)
        case 'user_dataset_store':
            return str(ENV.user_dataset_store)
        case 'user_test_store':
            return str(ENV.user_test_store)
        case 'user_delivery_store':
            return str(ENV.user_delivery_store)
        case 'user_log_store':
            return str(ENV.user_log_store)
        case 'config_store':
            return str(ENV.config_store)
        case 'artefact_store':
            return str(ENV.artefact_store)
        case 'model_store':
            return str(ENV.model_store)
        case 'dataset_store':
            return str(ENV.dataset_store)
        case 'test_store':
            return str(ENV.test_store)
        case 'delivery_store':
            return str(ENV.delivery_store)
        case 'log_store':
            return str(ENV.log_store)
        case _:
            raise ValueError(f'the store {store_name} is not a valid store name')


OmegaConf.register_new_resolver(name='odn.store', resolver=store_resolver)

__all__ = ['ENV', 'ODEON_ENV', 'ODEON_PATH']
