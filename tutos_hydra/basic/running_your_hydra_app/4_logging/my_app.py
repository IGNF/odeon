# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging

from omegaconf import DictConfig, OmegaConf

import hydra

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path=None, config_name=None)
def my_app(_cfg: DictConfig) -> None:
    log.info("Info level message")
    log.debug("Debug level message")
    log.info(f"Config Schema: \n {OmegaConf.to_container(_cfg)}")

if __name__ == "__main__":
    my_app()
