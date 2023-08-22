""" Env module, used to bind configuration at user level """
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .default_path import ODEON_PATH
from .io_utils import create_path_if_not_exists
from .logger import get_logger
# from .types import PARSER
from .singleton import Singleton
from .types import PARAMS, URI

logger = get_logger(__name__)
ENV_VARIABLE = 'env_variables'
ODEON_DEBUG_MODE_ENV_VARIABLE = 'ODEON_DEBUG_MODE'
DEFAULT_PARSER = 'omegaconf'

DEFAULT_CONFIG_STORE = ODEON_PATH / 'config_store'
DEFAULT_ARTEFACT_STORE = ODEON_PATH / 'artefact_store'

DEFAULT_FEATURE_STORE = ODEON_PATH / 'feature_store'
DEFAULT_DATASET_STORE = ODEON_PATH / 'dataset_store'

DEFAULT_MODEL_STORE = ODEON_PATH / 'model_store'
DEFAULT_TEST_STORE = ODEON_PATH / 'test_store'

DEFAULT_DELIVERY_STORE = ODEON_PATH / 'delivery_store'


def set_env_variables(variables: Dict):
    for k, v in variables.items():
        os.environ[str(k)] = str(v)
        logger.info(f'environment variable {k}: {os.environ[k]} has been set')


def is_env_variable_exists(variable: str) -> bool:
    return os.environ.get(variable) is not None


def get_env_variable(variable: str) -> Optional[Any]:
    return os.environ.get(variable)


def _debug_mode() -> bool:
    dev_mode = get_env_variable(ODEON_DEBUG_MODE_ENV_VARIABLE)
    if dev_mode:
        if int(str(dev_mode)) == 1:
            return True
        else:
            return False
    else:
        return False


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class Member:
    name: str = ''
    email: str = ''
    role: str = ''
    id: str = str(uuid4().hex)  # default id, randomly generated
    microsoft_teams_id: str = ''
    current_user: bool = False


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class User(Member):
    ...


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class Team:
    name: str = ''
    email: str = ''
    members: List[Member] | Dict[str, Member] | None = field(default_factory=lambda: [Member(current_user=True)])
    current_team: bool = True
    _reverse_dict: Optional[Dict[str, Member]] = field(default=None)


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class Project:
    name: str = ''
    email: str = ''
    teams: List[Team] | Dict[str, Team] | None = None


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class EnvConf:

    set_env_variables: Optional[PARAMS] = None
    get_env_variables: Optional[List[str]] = None
    config_parser: str = DEFAULT_PARSER
    config_store: URI = DEFAULT_CONFIG_STORE
    artefact_store: URI = DEFAULT_ARTEFACT_STORE
    feature_store: URI = DEFAULT_FEATURE_STORE
    dataset_store: URI = DEFAULT_DATASET_STORE
    model_store: URI = DEFAULT_MODEL_STORE
    test_store: URI = DEFAULT_TEST_STORE
    delivery_store: URI = DEFAULT_DELIVERY_STORE
    debug_mode: bool = field(default_factory=_debug_mode)
    user: Optional[User] = field(default=None)
    team: Optional[Team] = field(default=None)
    project: Optional[Project] = field(default=None)

    def __post_init__(self):

        if self.set_env_variables is not None:
            set_env_variables(variables=self.set_env_variables)

        create_path_if_not_exists(self.config_store)
        create_path_if_not_exists(self.artefact_store)
        create_path_if_not_exists(self.feature_store)
        create_path_if_not_exists(self.dataset_store)
        create_path_if_not_exists(self.model_store)
        create_path_if_not_exists(self.test_store)
        create_path_if_not_exists(self.delivery_store)


class Env(metaclass=Singleton):
    def __init__(self, config: EnvConf):

        self.config: EnvConf = config
    # TODO :build stores if not exists
