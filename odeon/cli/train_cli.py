import pytorch_lightning as pl
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.configs.core import Config
from odeon.utils.instantiate import (
    instantiate_datamodule,
    instantiate_module,
    instantiate_trainer
)


def train(config: Config)-> None:

    try:

        datamodule = instantiate_datamodule(config=config.datamodule, transform_config=config.transforms)

        module = instantiate_module(config=config, datamodule=datamodule)

        trainer = instantiate_trainer(config)

        trainer.fit(model=module, datamodule=datamodule)

        if config.run_test is True:
            trainer.test()
        elif config.run_pred is True:
            trainer.pred()

    except OdeonError as error:
        LOGGER.error("ERROR: Something went wrong during the fit step of the training")
        raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                            "ERROR: Something went wrong during the fit step of the training",
                            stack_trace=error)

    return 0

