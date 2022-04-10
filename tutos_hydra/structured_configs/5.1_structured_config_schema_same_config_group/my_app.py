# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass

from omegaconf import MISSING, OmegaConf
from typing import Any
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class DBConfig:
    driver: str = MISSING
    host: str = "localhost"
    port: int = MISSING


@dataclass
class MySQLConfig:
    driver: str = "mysql"
    port: int = 3306
    user: str = MISSING
    password: str = MISSING


@dataclass
class PostGreSQLConfig:
    driver: str = "postgresql"
    user: str = MISSING
    port: int = 5432
    password: str = MISSING
    timeout: int = 10


@dataclass
class Config:
    db: Any = MISSING
    debug: bool = False


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="db", name="base_mysql", node=MySQLConfig)
cs.store(group="db", name="base_postgresql", node=PostGreSQLConfig)


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: Config) -> None:
    print(60 * "#")
    print(OmegaConf.to_yaml(cfg))
    print(60 * "#")
    print(OmegaConf.to_container(cfg))
    print(60 * "#")


if __name__ == "__main__":
    my_app()
