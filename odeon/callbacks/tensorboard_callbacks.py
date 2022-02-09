import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from matplotlib import colors
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.nn.datasets import PatchDataset

NUM_PREDICTIONS = 5

def get_tensorboard_logger_idx(trainer):
    tensorboard_logger_idx = []
    for idx, logger in enumerate(trainer.logger.experiment):
        if isinstance(logger, torch.utils.tensorboard.writer.SummaryWriter):
            tensorboard_logger_idx.append(idx)
    return tensorboard_logger_idx


class GraphAdder(pl.Callback):

    def __init__(self, samples=None):
        super().__init__()
        self.samples = samples
        self.tensorboard_logger_idx = None
        self.samples = samples

    def on_fit_start(self, trainer, pl_module):
        self.tensorboard_logger_idx = get_tensorboard_logger_idx(trainer=trainer)
        model_device = next(iter(pl_module.model.parameters())).device
        if self.samples is None:
            self.samples = next(iter(trainer.datamodule.val_dataloader()))["image"]
        self.samples = self.samples.to(model_device)
        if len(self.tensorboard_logger_idx) == 1:
            logger_idx = [0]
            trainer.logger.experiment[logger_idx].add_graph(pl_module.model, self.samples)
        elif len(self.tensorboard_logger_idx) > 1:
            for logger_idx in self.tensorboard_logger_idx:
                trainer.logger.experiment[logger_idx].add_graph(pl_module.model, self.samples)
        else:
            LOGGER.error("ERROR: the callback GraphAdder won't work if there is any Tensorboard logger.")
            raise OdeonError(ErrorCodes.ERR_CALLBACK_ERROR,
                             "GraphAdder callback is not use properly.")
        # Detach samples from device
        self.samples = self.samples.detach()


class HistogramAdder(pl.Callback):
    
    def __init__(self):
        super().__init__()
        self.tensorboard_logger_idx = None

    def on_fit_start(self, trainer, pl_module):
        self.tensorboard_logger_idx = get_tensorboard_logger_idx(trainer=trainer)

    def add_histogram(self, pl_module, phase_index):
        if len(self.tensorboard_logger_idx) == 1: 
            for name, params in pl_module.named_parameters():
                pl_module.logger.experiment.add_histogram(name, params, pl_module.current_epoch)
        elif len(self.tensorboard_logger_idx) > 1:
            for name, params in pl_module.named_parameters():
                pl_module.logger.experiment[phase_index].add_histogram(name, params, pl_module.current_epoch)
        else:
            LOGGER.error("ERROR: the callback GraphAdder won't work if there is any Tensorboard logger.")
            raise OdeonError(ErrorCodes.ERR_CALLBACK_ERROR,
                             "GraphAdder callback is not use properly.")

    def on_train_epoch_end(self, trainer, pl_module):
        self.add_histogram(pl_module=pl_module, phase_index=0)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        self.add_histogram(pl_module=pl_module, phase_index=1)

    def on_test_epoch_end(self, trainer, pl_module):
        self.add_histogram(pl_module=pl_module, phase_index=2)


class PredictionsAdder(pl.Callback):

    def __init__(self, samples=None, num_predictions=NUM_PREDICTIONS):
        super().__init__()
        self.classes_colors =  {'batiment' : 'red', 
                                'zone_impermeable': 'navy',
                                'zone_permeable': 'hotpink',
                                'piscine': 'aqua',
                                'sol_nu': 'sandybrown',
                                'surface_eau': 'dodgerblue',
                                'neige': 'lavender',
                                'coupe': 'chartreuse',
                                'peuplement_feuillus': 'forestgreen',
                                'peuplement_coniferes': 'darkgreen',
                                'lande_ligneuse': 'palegreen',
                                'vigne': 'indigo',
                                'culture': 'lime',
                                'terre_arable': 'maroon',
                                'autre': 'black'}
        self.tensorboard_logger_idx = None
        self.samples = samples
        self.num_predictions = num_predictions
        self.sample_dataset = None

    def on_fit_start(self, trainer, pl_module):
        self.tensorboard_logger_idx = get_tensorboard_logger_idx(trainer=trainer)
        if self.samples is None:
            self.sample_dataset = PatchDataset(image_files=trainer.datamodule.val_image_files,
                                               mask_files=trainer.datamodule.val_mask_files,
                                               transform=None,
                                               image_bands=trainer.datamodule.image_bands,
                                               mask_bands=trainer.datamodule.mask_bands,
                                               width=trainer.datamodule.width,
                                               height=trainer.datamodule.height)
            self.sample_loader = DataLoader(dataset=self.sample_dataset,
                                            batch_size=self.num_predictions,
                                            num_workers=trainer.datamodule.num_workers,
                                            pin_memory=trainer.datamodule.pin_memory,
                                            shuffle=True)
            self.samples = next(iter(self.sample_loader))

    def add_predictions(self, trainer, pl_module, phase_index):
        model_device = next(iter(pl_module.model.parameters())).device
        images, targets = self.samples["image"].to(model_device), self.samples["mask"].to(model_device)
        with torch.no_grad():
            logits = pl_module.forward(images)
            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(proba, dim=1)
            targets = torch.argmax(targets, dim=1).type(torch.int32)
        cmap = colors.ListedColormap(self.classes_colors.values())
        grids = []
        for image, target, pred in zip(images, targets, preds):
            pred_rgba = torch.Tensor(cmap(pred.cpu().numpy()).swapaxes(0, 2).swapaxes(1, 2)[:3, :, :])
            target_rgba = torch.Tensor(cmap(target.cpu().numpy()).swapaxes(0, 2).swapaxes(1, 2)[:3, :, :])
            grids.append(make_grid([image.cpu(), target_rgba, pred_rgba]))
        image_grid = torch.cat(grids, 1)
        pl_module.logger.experiment[phase_index].add_image("Predictions", image_grid, pl_module.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        self.add_predictions(trainer=trainer, pl_module=pl_module, phase_index=0)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.add_predictions(trainer=trainer, pl_module=pl_module, phase_index=1)

    def on_test_epoch_end(self, trainer, pl_module):
        self.add_predictions(trainer=trainer, pl_module=pl_module, phase_index=2)
