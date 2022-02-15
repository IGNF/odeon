import os
from time import gmtime, strftime
from importlib_metadata import version
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from odeon.commons.exception import OdeonError, ErrorCodes


class MyModelCheckpoint(ModelCheckpoint):

    def __init__(self, monitor, dirpath, filename=None, version=None, **kwargs):
        if filename is None:
            filename = "checkpoint-{epoch:02d}-{" + monitor + ":.2f}"
        self.version = version
        dirpath = self.check_path_ckpt(dirpath)
        super().__init__(monitor=monitor, dirpath=dirpath, filename=filename, **kwargs)

    def check_path_ckpt(self, path): 
        if not os.path.exists(path):
            path_ckpt = path if self.version is None else os.path.join(path, self.version)
        else:
            description = "version_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
            path_ckpt = os.path.join(path, description)
        return path_ckpt


class HistorySaver(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        if pl_module.logger is not None:
            idx_csv_loggers = [idx for idx, logger in enumerate(pl_module.logger.experiment)\
                if isinstance(logger, pl.loggers.csv_logs.ExperimentWriter)]
            self.idx_loggers = {'val': idx_csv_loggers[0], 'test': idx_csv_loggers[-1]}

    def on_validation_epoch_end(self, trainer, pl_module):
        logger_idx = self.idx_loggers['val']
        metric_collection = pl_module.val_epoch_metrics.copy()
        metric_collection['loss'] = pl_module.val_epoch_loss
        metric_collection['learning rate'] = pl_module.learning_rate  # Add learning rate logging  
        pl_module.logger.experiment[logger_idx].log_metrics(metric_collection, pl_module.current_epoch)
        pl_module.logger.experiment[logger_idx].save()

# Check size of tensors in forward pass
class CheckBatchGradient(pl.Callback):
    
    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()
        
        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)
        
        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")