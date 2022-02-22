import os
import json
from time import gmtime, strftime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from odeon import LOGGER
from odeon.commons.guard import files_exist
from odeon.nn.models import get_train_filenames, save_model
from odeon.commons.exception import OdeonError, ErrorCodes


class MyModelCheckpoint(ModelCheckpoint):

    def __init__(self,
                 monitor,
                 dirpath,
                 filename=None,
                 version=None,
                 **kwargs):

        if filename is None:
            filename = "checkpoint-{epoch:02d}-{" + monitor + ":.2f}"
        self.version = version
        dirpath = self.check_path_ckpt(dirpath)
        super().__init__(monitor=monitor, dirpath=dirpath, filename=filename, **kwargs)

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

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)


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

    def on_test_epoch_end(self, trainer, pl_module):
        logger_idx = self.idx_loggers['test']
        metric_collection = pl_module.test_epoch_metrics.copy()
        metric_collection['loss'] = pl_module.test_epoch_loss
        pl_module.logger.experiment[logger_idx].log_metrics(metric_collection, pl_module.current_epoch)
        pl_module.logger.experiment[logger_idx].save()


class ContinueTraining(pl.Callback):
    def __init__(self,
                 out_dir,
                 out_filename,
                 save_history=False):
        super().__init__()
        self.out_dir = out_dir
        self.out_filename = out_filename
        self.save_history = save_history
        self.train_files = get_train_filenames(self.out_dir, self.out_filename)
        check_train_files = [self.train_files["model"], self.train_files["optimizer"]]
        files_exist(check_train_files)
        self.history_dict = None

    def on_fit_start(self, trainer, pl_module):
        current_device = next(iter(pl_module.model.parameters())).device
        model_state_dict = torch.load(self.train_files["model"],
                                      map_location=current_device)
        pl_module.model.load_state_dict(state_dict=model_state_dict)

        optimizer_state_dict = torch.load(self.train_files["optimizer"],
                                          map_location=current_device)

        pl_module.optimizer.load_state_dict(state_dict=optimizer_state_dict)

        if Path(self.train_files["history"]).exists():
            # Recuperation epoch and learning rate to resume the training
            try:
                with open(self.train_files["history"], 'r') as file:
                    self.history_dict = json.load(file)
            except OdeonError as error:
                raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                                 f"{self.train_files['history']} not found",
                                 stack_trace=error)

        if Path(self.train_files["train"]).exists():
            train_dict = torch.load(self.train_files["train"])
            pl_module.scheduler.load_state_dict(train_dict["scheduler"])
        return super().on_fit_start(trainer, pl_module)


class ExoticCheckPoint(pl.Callback):

    def __init__(self, out_folder, out_filename, model_out_ext):
        super().__init__()
        self.out_folder = out_folder
        self.model_out_ext = model_out_ext
        if os.path.splitext(out_filename)[-1] != self.model_out_ext:
            self.out_filename = os.path.splitext(out_filename)[0] + self.model_out_ext
        else:
            self.out_filename = out_filename
        self.best_val_loss = None
        self.input_sample = None

    def on_fit_start(self, trainer, pl_module):
        # if self.model_out_ext == ".onnx":
        #     self.sample_loader = DataLoader(dataset=trainer.datamodule.val_dataset,
        #                                     batch_size=trainer.datamodule.train_batch_size,
        #                                     num_workers=trainer.datamodule.num_workers,
        #                                     pin_memory=trainer.datamodule.pin_memory)
        #     self.input_sample = next(iter(self.sample_loader))["image"]
        return super().on_fit_start(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.compare_and_save(trainer, pl_module)

    def on_exception(self, trainer, pl_module, exception):
        #TODO will not find the val_loss of the current epoch..
        self.compare_and_save(trainer, pl_module)
        return super().on_exception(trainer, pl_module, exception)

    def compare_and_save(self, trainer, pl_module):
        if self.best_val_loss is None:  
            self.best_val_loss = pl_module.val_epoch_loss

        elif self.best_val_loss < pl_module.val_epoch_loss:
    
            if self.out_filename is None:
                self.out_filename =  f"checkpoint-epoch{pl_module.current_epoch}-val_loss{pl_module.val_epoch_loss}"

            self. best_val_loss = pl_module.val_epoch_loss

            if self.model_out_ext == ".pth":
                model_filepath = save_model(out_dir=self.out_folder,
                                            out_filename=self.out_filename,
                                            model=pl_module.model, 
                                            optimizer=pl_module.optimizer,
                                            scheduler=pl_module.scheduler)

            # elif self.model_out_ext == ".onnx":
            #     model_filepath = os.path.join(self.out_folder, self.out_filename)
            #     model_device = next(iter(pl_module.model.parameters())).device
            #     self.input_sample = self.input_sample.to(model_device)
            #     torch.onnx.export(pl_module.model,
            #                       self.input_sample,
            #                       model_filepath,
            #                       export_params=True,        # store the trained parameter weights inside the model file
            #                       do_constant_folding=True,  # whether to execute constant folding for optimization
            #                       input_names = ['input'],   # the model's input names
            #                       output_names = ['output'], # the model's output names
            #                       dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
            #                                     'output' : {0 : 'batch_size'}})
            #     self.input_sample.detach()
            LOGGER.info(f"Saving {model_filepath}")


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
