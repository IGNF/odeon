# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(60 * "#")
    print(OmegaConf.to_yaml(cfg))
    print(60 * "#")
    print(OmegaConf.to_container(cfg))
    print(60 * "#")

if __name__ == "__main__":
    my_app()
