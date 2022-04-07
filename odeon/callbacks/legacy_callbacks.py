import os
import json
from pathlib import Path
import torch
# from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from odeon import LOGGER
from odeon.commons.guard import files_exist
from odeon.nn.models import get_train_filenames, save_model
from odeon.commons.exception import OdeonError, ErrorCodes


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

    @rank_zero_only
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

    def __init__(self, out_dir, out_filename, model_out_ext):
        super().__init__()
        self.out_dir = out_dir
        self.model_out_ext = model_out_ext
        if os.path.splitext(out_filename)[-1] != self.model_out_ext:
            self.out_filename = os.path.splitext(out_filename)[0] + self.model_out_ext
        else:
            self.out_filename = out_filename
        self.best_val_loss = None
        self.input_sample = None

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        # if self.model_out_ext == ".onnx":
        #     self.sample_loader = DataLoader(dataset=trainer.datamodule.val_dataset,
        #                                     batch_size=trainer.datamodule.train_batch_size,
        #                                     num_workers=trainer.datamodule.num_workers,
        #                                     pin_memory=trainer.datamodule.pin_memory)
        #     self.input_sample = next(iter(self.sample_loader))["image"]
        return super().on_fit_start(trainer, pl_module)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        self.compare_and_save(trainer, pl_module)

    @rank_zero_only
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
                model_filepath = save_model(out_dir=self.out_dir,
                                            out_filename=self.out_filename,
                                            model=pl_module.model, 
                                            optimizer=pl_module.optimizer,
                                            scheduler=pl_module.scheduler)

            # elif self.model_out_ext == ".onnx":
            #     model_filepath = os.path.join(self.out_dir, self.out_filename)
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