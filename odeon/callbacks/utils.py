import os
from time import gmtime, strftime 
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from odeon.loggers.json_logs import JSONLogger
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes

THRESHOLD = 0.5


class LightningCheckpoint(ModelCheckpoint):

    def __init__(
        self,
        monitor,
        dirpath,
        save_top_k,
        filename=None,
        version=None,
        **kwargs,
        ):

        self.save_top_k = save_top_k
        if filename is None:
            filename = "checkpoint-{epoch:02d}-{" + monitor + ":.2f}"
        elif self.save_top_k > 1:
            filename = os.path.splitext(filename)[0] + "-{epoch:02d}-{" + monitor + ":.2f}"
        else:
            filename = os.path.splitext(filename)[0]

        self.version = version
        dirpath = self.check_path_ckpt(dirpath)
        super().__init__(monitor=monitor, dirpath=dirpath, filename=filename, save_top_k=save_top_k, **kwargs)

    def check_path_ckpt(self, path):
        if not os.path.exists(path):
            path_ckpt = path if self.version is None else os.path.join(path, self.version)
        else:
            if self.version is None:
                description = "version_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
            else:
                description = self.version
            path_ckpt = os.path.join(path, description)
        return path_ckpt

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        return super().on_load_checkpoint(trainer, pl_module, callback_state)

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)


class HistorySaver(pl.Callback):
    
    def __init__(self):
        super().__init__()
        self.idx_json_loggers = None
        self.phase_dict = {'val': 0,
                           'test': 1}

    def get_json_logger(self, trainer: Trainer, phase: str) -> JSONLogger:
        """
            Safely get JSONlogger from Trainer attributes according to the current phase.
        """
        if self.idx_json_loggers is None:
            self.idx_json_loggers = []

            if isinstance(trainer.logger, JSONLogger):
                self.idx_json_loggers = 0

            elif isinstance(trainer.logger, LoggerCollection):
                for idx, logger in enumerate(trainer.logger):
                    if isinstance(logger, JSONLogger):
                        self.idx_json_loggers.append(idx)

        if self.idx_json_loggers:
            if self.idx_json_loggers == 0:
                return trainer.logger
            else:
                phase_idx = self.phase_dict[phase]
                logger_idx = self.idx_json_loggers[phase_idx]
                return trainer.logger[logger_idx]
        else:
            LOGGER.error("ERROR: the callback HistogramAdder won't work if there is any logger of type JSONLogger.")
            raise OdeonError(ErrorCodes.ERR_CALLBACK_ERROR,
                             "HistogramAdder callback is not use properly.")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        logger = self.get_json_logger(trainer=trainer, phase='val')
        metric_collection = {key: value.cpu().numpy() for key, value in pl_module.val_epoch_metrics.items()}
        metric_collection['loss'] = pl_module.val_epoch_loss.cpu().numpy()
        metric_collection['learning rate'] = pl_module.hparams.learning_rate  # Add learning rate logging  
        logger.experiment.log_metrics(metric_collection, pl_module.current_epoch)
        logger.experiment.save()

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        logger = self.get_json_logger(trainer=trainer, phase='test')
        metric_collection = {key: value.cpu().numpy() for key, value in pl_module.test_epoch_metrics.items()}
        metric_collection['loss'] = pl_module.test_epoch_loss.cpu().numpy()
        logger.experiment.log_metrics(metric_collection, pl_module.current_epoch)
        logger.experiment.save()
