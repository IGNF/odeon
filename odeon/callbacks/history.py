import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection
from pytorch_lightning.utilities import rank_zero_only

from odeon import LOGGER
from odeon.commons.exception import ErrorCodes, OdeonError
from odeon.loggers.json_logs import JSONLogger

THRESHOLD = 0.5


class HistorySaver(pl.Callback):
    def __init__(self):
        super().__init__()
        self.idx_json_loggers = None
        self.phase_dict = {"val": 0, "test": 1}

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
            LOGGER.error(
                "ERROR: the callback HistogramAdder won't work if there is any logger of type JSONLogger."
            )
            raise OdeonError(
                ErrorCodes.ERR_CALLBACK_ERROR,
                "HistogramAdder callback is not use properly.",
            )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        logger = self.get_json_logger(trainer=trainer, phase="val")
        metric_collection = {
            key: value.cpu().numpy()
            for key, value in pl_module.val_epoch_metrics.items()
        }
        metric_collection["loss"] = pl_module.val_epoch_loss.cpu().numpy()
        metric_collection[
            "learning rate"
        ] = pl_module.hparams.learning_rate  # Add learning rate logging
        logger.experiment.log_metrics(metric_collection, pl_module.current_epoch)
        logger.experiment.save()

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        logger = self.get_json_logger(trainer=trainer, phase="test")
        metric_collection = {
            key: value.cpu().numpy()
            for key, value in pl_module.test_epoch_metrics.items()
        }
        metric_collection["loss"] = pl_module.test_epoch_loss.cpu().numpy()
        logger.experiment.log_metrics(metric_collection, pl_module.current_epoch)
        logger.experiment.save()
