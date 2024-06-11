""" Env module, used to bind configuration at user level """
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from jsonargparse import ArgumentParser, set_config_read_mode

from .default_path import ODEON_ENV, ODEON_PATH
from .io_utils import create_path_if_not_exists, generate_yaml_with_doc
from .logger import get_logger
from .plugins.plugin import OdnPlugin, load_plugins
from .python_env import debug_mode
# from .logger introspection.py get_logger
# from .types introspection.py PARSER
from .singleton import Singleton
from .types import PARAMS, PARSER, URI

# enable URL with Parser
_URL_ENABLED: bool = True
set_config_read_mode(urls_enabled=_URL_ENABLED)  # enable URL as path file

logger = get_logger(__name__, debug=debug_mode)
ENV_VARIABLE = 'env_variables'
ODEON_DEBUG_MODE_ENV_VARIABLE = 'ODEON_DEBUG_MODE'

DEFAULT_PARSER: PARSER = 'omegaconf'
ODEON_PARSE_ENV_VARIABLE = 'ODEON_PARSER'
ODEON_PARSER_AVAILABLE: List[PARSER] = ['yaml', 'jsonnet', 'omegaconf']

DEFAULT_CONFIG_STORE = ODEON_PATH / 'config_store'
DEFAULT_ARTEFACT_STORE = ODEON_PATH / 'artefact_store'
DEFAULT_FEATURE_STORE = ODEON_PATH / 'feature_store'
DEFAULT_DATASET_STORE = ODEON_PATH / 'dataset_store'

DEFAULT_MODEL_STORE = ODEON_PATH / 'model_store'
DEFAULT_TEST_STORE = ODEON_PATH / 'test_store'

DEFAULT_LOG_STORE = ODEON_PATH / 'log_store'
DEFAULT_DELIVERY_STORE = ODEON_PATH / 'delivery_store'

DEFAULT_PLUGIN_CONF = {'albu_transform': 'odeon.data:albu_transform_plugin',
                       'data': 'odeon.data:data_plugin',
                       'model': 'odeon.models:model_plugin',
                       'fit': 'odeon.fit:fit_plugin',
                       'pl_logger': 'odeon.fit:pl_logger_plugin',
                       'pl_callback': 'odeon.fit:pl_callback_plugin',
                       'binary_metric': 'odeon.metrics:binary_metric_plugin',
                       'multiclass_metric': 'odeon.metrics:multiclass_metric_plugin',
                       'multilabel_metric': 'odeon.metrics:multilabel_metric_plugin',
                       'app_plugin': 'odeon.core:app_plugin'}

ENV_PREFIX: str = 'env'

DEFAULT_ENV_CONF: Dict[str, PARAMS] = {ENV_PREFIX: {'plugins': DEFAULT_PLUGIN_CONF,
                                                    'model_store': str(DEFAULT_MODEL_STORE),
                                                    'test_store': str(DEFAULT_TEST_STORE),
                                                    'delivery_store': str(DEFAULT_DELIVERY_STORE),
                                                    'dataset_store': str(DEFAULT_DATASET_STORE),
                                                    'artefact_store': str(DEFAULT_ARTEFACT_STORE),
                                                    'config_store': str(DEFAULT_CONFIG_STORE),
                                                    'feature_store': str(DEFAULT_FEATURE_STORE),
                                                    'log_store': str(DEFAULT_LOG_STORE)}}

DOC_ENV_CONF = '''
                Here is a complete example of how to configure an Odeon environment with your env.yml.
                By default, your env.yml will be loaded from $HOME, but you can also
                configure your path with the following environment variable: ODEON_INSTALL_PATH.


                set_env_variables:
                  API_KEY: "abc123"
                  ANOTHER_VAR: "xyz789"
                get_env_variables:
                  - "PATH"
                  - "HOME"
                config_parser: "omegaconf"
                config_store: "file:///path/to/config/store"
                artefact_store: "file:///path/to/artefact/store"
                feature_store: "file:///path/to/feature/store"
                dataset_store: "file:///path/to/dataset/store"
                model_store: "file:///path/to/model/store"
                test_store: "file:///path/to/test/store"
                delivery_store: "file:///path/to/delivery/store"
                log_store: "file:///path/to/log/store"
                debug_mode: true
                user:
                    name: "John Doe"
                    email: "john.doe@example.com"
                    role: "Data Scientist"
                    id: "123456"
                    microsoft_teams_id: "78910"
                    current_user: true
                team:
                    name: "Data Team"
                    email: "data.team@example.com"
                members:
                    - name: "John Doe"
                    email: "john.doe@example.com"
                    role: "Data Scientist"
                    id: "123456"
                    microsoft_teams_id: "78910"
                    current_user: true
                    current_team: true
                project:
                    name: "Remote Sensing Project"
                    email: "project@example.com"
                plugins: 'albu_transform': 'odeon.data:albu_transform_plugin',
                         'data': 'odeon.data:data_plugin',
                         'model': 'odeon.models:model_plugin',
                         'fit': 'odeon.fit:fit_plugin',
                         'pl_logger': 'odeon.fit.pl_logger_plugin',
                         'pl_callback': 'odeon.fit.pl_callback_plugin',
                         'binary_metric': 'odeon.metrics.binary_metric_plugin',
                         'multiclass_metric': 'odeon.metrics:multiclass_metric_plugin',
                         'multilabel_metric': 'odeon.metrics:multilabel_metric_plugin',
                         'app_plugin': 'odeon.core.app_plugin'
                '''


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
    log_store: URI = DEFAULT_LOG_STORE
    debug_mode: bool = debug_mode
    user: Optional[User] = None
    team: Optional[Team] = None
    project: Optional[Project] = None
    plugins: Dict[str, OdnPlugin | str] | List[OdnPlugin | str] | str | OdnPlugin | None = None


class Env(metaclass=Singleton):
    def __init__(self, config: Optional[EnvConf] = None):

        config = config if config is not None else EnvConf()
        self.set_env_variables: Optional[PARAMS] = None
        self.get_env_variables: Optional[List[str]] = None
        self.config_parser: str = DEFAULT_PARSER
        self.project: Project | None = config.project
        self.team: Team | None = config.team
        self.user: User | None = config.user
        self.plugins: Dict[str, OdnPlugin | str] | List[OdnPlugin | str] | str | OdnPlugin | None = config.plugins
        self.debug_mode: bool = config.debug_mode
        if self.set_env_variables is not None:
            set_env_variables(variables=self.set_env_variables)
        self.config_store: URI = config.config_store
        self.model_store: URI = config.model_store
        self.feature_store: URI = config.feature_store
        self.artefact_store: URI = config.artefact_store
        self.test_store: URI = config.test_store
        self.log_store: URI = config.log_store
        self.delivery_store: URI = config.delivery_store
        self.dataset_store: URI = config.dataset_store

        self.user_path: URI | None = self._create_user_path()
        self.user_config_store = Path(self.config_store) / self.user_path if self.user_path is not None\
            else Path(self.config_store)
        self.user_artefact_store = Path(self.artefact_store) / self.user_path if self.user_path is not None\
            else Path(self.artefact_store)
        self.user_feature_store = Path(self.feature_store) / self.user_path if self.user_path is not None\
            else Path(self.feature_store)
        self.user_dataset_store = Path(self.dataset_store) / self.user_path if self.user_path is not None\
            else Path(self.dataset_store)
        self.user_model_store = Path(config.model_store) / self.user_path if self.user_path is not None\
            else Path(self.model_store)
        self.user_test_store = Path(self.test_store) / self.user_path if self.user_path is not None\
            else Path(self.test_store)
        self.user_delivery_store = Path(config.delivery_store) / self.user_path if self.user_path is not None\
            else Path(config.delivery_store)
        self.user_log_store = Path(config.log_store) / self.user_path if self.user_path is not None\
            else Path(config.log_store)

        create_path_if_not_exists(self.user_config_store)
        create_path_if_not_exists(self.user_artefact_store)
        create_path_if_not_exists(self.user_feature_store)
        create_path_if_not_exists(self.user_dataset_store)
        create_path_if_not_exists(self.user_model_store)
        create_path_if_not_exists(self.user_test_store)
        create_path_if_not_exists(self.user_delivery_store)
        create_path_if_not_exists(self.user_log_store)

    def _create_user_path(self) -> URI | None:
        p: URI | None = None
        if self.project is not None:
            if self.project.name != '':
                p = Path(str(self.project.name))
        if self.team is not None:
            if self.team.name != '':
                p = Path(str(self.team.name)) if p is None else Path(p) / str(self.team.name)
        if self.user is not None:
            if self.user.name != '':
                p = Path(str(self.user.name)) if p is None else Path(p) / str(self.user.name)
        return p

    def load_plugins(self):

        load_plugins(self.plugins, force_registry=False)

    def reload_plugins(self):
        load_plugins(self.plugins, force_registry=True)


ENV: Optional[Env] = None


def get_env() -> Env:
    """
    Used to check if the .odeon directory is created and env.yml is inside.
    Otherwise, it will create the necessary directory and config file associated

    Returns
    -------

    Env
    """

    if ENV is None:
        if ODEON_ENV.is_file() is False:
            create_path_if_not_exists(ODEON_PATH)
            generate_yaml_with_doc(config_d=DEFAULT_ENV_CONF, docstring=DOC_ENV_CONF, filename=str(ODEON_ENV))
            # env_fields = fields(EnvConf())
            # end_d = {field.name: field.value}
            """"
            schema = OmegaConf.structured(EnvConf)
            conf = OmegaConf.load(ODEON_ENV)
            conf = OmegaConf.merge(schema, conf)
            assert isinstance(conf, Mapping)
            env_conf: EnvConf = EnvConf(**conf)
            """

        parser = ArgumentParser()
        parser.add_dataclass_arguments(theclass=EnvConf, nested_key='--env')
        cfg = parser.parse_path(str(ODEON_ENV))
        logger.debug(f'Config: {cfg}')
        instantiated_conf = parser.instantiate_classes(cfg=cfg)
        logger.debug(f'instanciated conf: {instantiated_conf}')
        conf_env = instantiated_conf.env
        logger.debug(f'instanciated conf env: {conf_env}')
        env = Env(config=conf_env)
        logger.debug(f'env: {env.plugins}')
        return env
    else:
        return ENV
