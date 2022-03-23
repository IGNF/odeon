import os
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
from pytorch_lightning import Trainer
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.commons.guard import dirs_exist, file_exist
from odeon.modules.datamodule import SegDataModule
from odeon.modules.seg_module import SegmentationTask
from odeon.nn.models import model_list
from odeon.configs.seg_config import OCSGEConfig

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_training")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)


cs = ConfigStore.instance()
cs.store(name="ocsge_config", node=OCSGEConfig)

@hydra.main(config_path="../configs", config_name="conf")
def main(cfg):

    print(cfg)

    return

    data_module = SegDataModule(cfg.datamodule)
    seg_module = SegmentationTask(cfg.taskmodule)
    trainer = Trainer(cfg.trainer)

    try:
        STD_OUT_LOGGER.info(
            f"Training : \n" 
            f"device: {cfg.trainer.device} \n"
            f"model: {cfg.task_module.model_name} \n"
            f"model file: {cfg.task_module.model_filename} \n"
            f"number of classes: {cfg.data_module.num_classes} \n"
            f"number of samples: {len(cfg.data_module.train_image_files) + len(cfg.data_module.val_image_files)}  "
            f"(train: {len(cfg.data_module.train_image_files)}, val: {len(cfg.data_module.val_image_files)})"
            )
        trainer.fit(model=seg_module,
                    datamodule=data_module)

    except OdeonError as error:
        raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                            "ERROR: Something went wrong during the fit step of the training",
                            stack_trace=error)

    return 0


if __name__ == "__main__":
    print("begining main ...")
    main()
    # trainer = TrainCLI()
