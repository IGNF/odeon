from typing import List, Optional, Union
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.profiler import BaseProfiler
from odeon.configs.core import Config
from odeon.modules.datamodule import SegDataModule
from odeon.modules.seg_module import SegmentationTask
from odeon.configs.core import DataModuleConf, TransformsConf


def instantiate_datamodule(
    config: DataModuleConf,
    transform_config: TransformsConf
    ) -> SegDataModule:
    """
        Instantiate pytorch lightning datamodule from Hydra config.
    """
    return SegDataModule(config, transform_config)


def instantiate_module(
    config: Config,
    datamodule: SegDataModule
    ) -> SegmentationTask:
    """
        Instantiate pytorch lightning module from Hydra config.
    """
    OmegaConf.set_struct(config, False)
    
    config.model.in_channels = datamodule.num_channels
    config.model.classes = datamodule.num_classes
    config.class_labels = datamodule.class_labels

    OmegaConf.set_struct(config, True)

    return SegmentationTask(config)


def instantiate_trainer(config: Config)-> Trainer:
    """Instantiate pytorch lightning trainer from Hydra config."""

    def _instantiate_callbacks(callbacks_config: Optional[DictConfig]=None) -> List[Callback]:
        """Instantiate pytorch lightning callbacks from Hydra config."""
        callbacks = [instantiate(callback_config) for callback_config in callbacks_config.values()]
        return callbacks

    def _instantiate_logger(loggers_config: Optional[DictConfig]=None) -> List[LightningLoggerBase]:
        """Instantiate pytorch lightning logger from Hydra config."""
        loggers = [instantiate(logger_config) for logger_config in loggers_config.values()]
        return loggers

    def _instantiate_profiler(profiler_config: Optional[DictConfig] = None) -> Optional[BaseProfiler]:
        """Instantiate pytorch lightning profiler from Hydra config."""
        return instantiate(profiler_config)

    trainer = Trainer(
        logger=_instantiate_logger(config.logger),
        profiler=_instantiate_profiler(config.profiler),
        callbacks=_instantiate_callbacks(config.callbacks),
        **config.trainer,
    )

    return trainer
