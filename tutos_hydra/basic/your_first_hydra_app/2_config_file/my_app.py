# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class PostgresSQLConfig:
	driver: str = "postgresql"
	user: str = 2
	password: str= "steph2001"

cs = ConfigStore.instance()
cs.store(group="db", name="postgre", node=PostgresSQLConfig)
cs.store(name="config2", node=PostgresSQLConfig(driver="test.db", user=3307), group="db")
# Using a dictionary, forfeiting runtime type safety
cs.store(name="config3", node={"host": "localhost", "port": 3308}, group="db")
1
@hydra.main(config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(60 * "#")
    print(OmegaConf.to_yaml(cfg))
    print(60 * "#")
    print(OmegaConf.to_container(cfg))
    print(60 * "#")

if __name__ == "__main__":
    my_app()
