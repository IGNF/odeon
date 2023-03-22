import os
from logging import getLogger
from typing import Dict

from odeon.core.env import Env, EnvConf, set_env_variables

logger = getLogger(__name__)


def test_env():
    env_var_default: Dict = {'test_0': 'test_0'}
    env_conf: EnvConf = EnvConf(env_variables=env_var_default)
    env: Env = Env(config=env_conf)
    logger.info(f'env: {env}')
    env_var: Dict = {'test_1': 'value_1',
                     'test_2': 'value_2'}
    set_env_variables(variables=env_var)
    for k, v in env_var.items():
        assert os.environ[k] == v
