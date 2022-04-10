import warnings
warnings.filterwarnings("ignore")
import pytorch_lightning as pl
from odeon import LOGGER, configs
from typing import Optional
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.configs.core import TrainConfig
import hydra
from omegaconf import OmegaConf 
from odeon.utils.print import print_config
from odeon.utils.instantiate import (
    instantiate_datamodule,
    instantiate_module,
    instantiate_trainer
)
from omegaconf import OmegaConf
from odeon.configs import database_lib
import hydra

CONFIG_PATH = "../configs/conf"  # Path of the directory where the config files are stored
CONFIG_NAME = "config" # basename of the file used for the config (here config.yaml which is in ../configs/conf/)


database_lib.register_configs()

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(config: TrainConfig)-> None:

    try:

        if config.print_config is True:
            print_config(config)

        if config.deterministic is True:
            pl.seed_everything(config.seed, workers=True)

        datamodule = instantiate_datamodule(config=config.datamodule, transform_config=config.transforms)

        module = instantiate_module(config=config, datamodule=datamodule)

        trainer = instantiate_trainer(config)

        trainer.fit(model=module, datamodule=datamodule)

        if config.run_test is True:
            trainer.test()
        elif config.run_pred is True:
            trainer.pred()

    except OdeonError as error:
        raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                            "ERROR: Something went wrong during the fit step of the training",
                            stack_trace=error)

    return 0


if __name__ == "__main__":
    main()
