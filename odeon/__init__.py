"""odeon package
"""
import pathlib
from pathlib import Path

from omegaconf import OmegaConf

from .core.default_path import ODEON_ENV, ODEON_PATH
from .core.env import Env, get_env
from .core.logger import get_logger
from .core.python_env import debug_mode

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
    store_mapping = {
        'user_config': str(ENV.user_config_store),
        'user_artefact': str(ENV.user_artefact_store),
        'user_model': str(ENV.user_model_store),
        'user_dataset': str(ENV.user_dataset_store),
        'user_test': str(ENV.user_test_store),
        'user_delivery': str(ENV.user_delivery_store),
        'user_log': str(ENV.user_log_store),
        'config': str(ENV.config_store),
        'artefact': str(ENV.artefact_store),
        'model': str(ENV.model_store),
        'dataset': str(ENV.dataset_store),
        'test': str(ENV.test_store),
        'delivery': str(ENV.delivery_store),
        'log': str(ENV.log_store)
    }
    if store_name in store_mapping:
        return store_mapping[store_name]
    else:
        raise ValueError(f'the store {store_name} is not a valid store name')


OmegaConf.register_new_resolver(name='odn.store', resolver=store_resolver)

__all__ = ['ENV', 'ODEON_ENV', 'ODEON_PATH']
