from random import sample
from cv2 import phase
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, draw_segmentation_masks
import pytorch_lightning as pl
from matplotlib import colors
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.nn.datasets import PatchDataset

NUM_PREDICTIONS = 5
OCSGE_LUT = [
 (219,  14, 154),
 (114, 113, 112),
 (248,  12,   0),
 ( 61, 230, 235),
 (169, 113,   1),
 ( 21,  83, 174),
 (255, 255, 255),
 (138, 179, 160),
 ( 70, 228, 131),
 ( 25,  74,  38),
 (243, 166,  13),
 (102,   0, 130),
 (255, 243,  13),
 (228, 223, 124),
 (  0,   0,   0)
]

def get_tensorboard_logger_idx(trainer):
    tensorboard_logger_idx = []
    for idx, logger in enumerate(trainer.logger.experiment):
        if isinstance(logger, torch.utils.tensorboard.writer.SummaryWriter):
            tensorboard_logger_idx.append(idx)
    return tensorboard_logger_idx

def map_phase_logger_idx(phase):
    phase_table = {"train": 0,
                   "val": 1,
                   "test": 2}
    assert phase in phase_table.keys(), "Phases are train/val/test."
    return phase_table[phase]


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

    def add_histogram(self, pl_module, phase):
        if len(self.tensorboard_logger_idx) == 1: 
            for name, params in pl_module.named_parameters():
                pl_module.logger.experiment.add_histogram(name, params, pl_module.current_epoch)
        elif len(self.tensorboard_logger_idx) > 1:
            phase_index = self.tensorboard_logger_idx[map_phase_logger_idx(phase)]
            for name, params in pl_module.named_parameters():
                pl_module.logger.experiment[phase_index].add_histogram(name, params, pl_module.current_epoch)
        else:
            LOGGER.error("ERROR: the callback GraphAdder won't work if there is any Tensorboard logger.")
            raise OdeonError(ErrorCodes.ERR_CALLBACK_ERROR,
                             "GraphAdder callback is not use properly.")

    def on_train_epoch_end(self, trainer, pl_module):
        self.add_histogram(pl_module=pl_module, phase='train')

    def on_validation_epoch_end(self, trainer, pl_module):
        self.add_histogram(pl_module=pl_module, phase='val')

    def on_test_epoch_end(self, trainer, pl_module):
        self.add_histogram(pl_module=pl_module, phase='test')


class PredictionsAdder(pl.Callback):

    def __init__(self, train_samples=None, val_samples=None, test_samples=None, num_predictions=NUM_PREDICTIONS):
        super().__init__()
        self.tensorboard_logger_idx = None
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.num_predictions = num_predictions
        self.sample_dataset = None

    def on_fit_start(self, trainer, pl_module):
        self.tensorboard_logger_idx = get_tensorboard_logger_idx(trainer=trainer)
        if self.train_samples is None:
            self.train_sample_dataset = PatchDataset(image_files=trainer.datamodule.train_image_files,
                                                     mask_files=trainer.datamodule.train_mask_files,
                                                     transform=None,
                                                     image_bands=trainer.datamodule.image_bands,
                                                     mask_bands=trainer.datamodule.mask_bands,
                                                     width=trainer.datamodule.width,
                                                     height=trainer.datamodule.height)
            self.train_sample_loader = DataLoader(dataset=self.train_sample_dataset,
                                                  batch_size=self.num_predictions,
                                                  num_workers=trainer.datamodule.num_workers,
                                                  pin_memory=trainer.datamodule.pin_memory,
                                                  shuffle=True)
            self.train_samples = next(iter(self.train_sample_loader))

        if self.val_samples is None:
            self.val_sample_dataset = PatchDataset(image_files=trainer.datamodule.val_image_files,
                                                   mask_files=trainer.datamodule.val_mask_files,
                                                   transform=None,
                                                   image_bands=trainer.datamodule.image_bands,
                                                   mask_bands=trainer.datamodule.mask_bands,
                                                   width=trainer.datamodule.width,
                                                   height=trainer.datamodule.height)
            self.val_sample_loader = DataLoader(dataset=self.val_sample_dataset,
                                                batch_size=self.num_predictions,
                                                num_workers=trainer.datamodule.num_workers,
                                                pin_memory=trainer.datamodule.pin_memory,
                                                shuffle=True)
            self.val_samples = next(iter(self.val_sample_loader))

    def on_test_start(self, trainer, pl_module):
        if self.test_samples is None:
            self.test_sample_dataset = PatchDataset(image_files=trainer.datamodule.test_image_files,
                                                    mask_files=trainer.datamodule.test_mask_files,
                                                    transform=None,
                                                    image_bands=trainer.datamodule.image_bands,
                                                    mask_bands=trainer.datamodule.mask_bands,
                                                    width=trainer.datamodule.width,
                                                    height=trainer.datamodule.height)
            self.test_sample_loader = DataLoader(dataset=self.test_sample_dataset,
                                                 batch_size=self.num_predictions,
                                                 num_workers=trainer.datamodule.num_workers,
                                                 pin_memory=trainer.datamodule.pin_memory,
                                                 shuffle=True)
            self.test_samples = next(iter(self.test_sample_loader))

    def add_predictions(self, trainer, pl_module, phase):
        if phase == "train":
            samples = self.train_samples
        elif phase == "val":
            samples = self.val_samples
        elif phase == "test":
            samples = self.test_samples

        model_device = next(iter(pl_module.model.parameters())).device
        images, targets = samples["image"].to(model_device), samples["mask"].to(model_device)
        with torch.no_grad():
            logits = pl_module.forward(images)
            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(proba, dim=1)

        images = images.cpu().type(torch.uint8)
        targets = targets.cpu()
        preds = preds.cpu()
        grids = []
        for image, target, pred in zip(images, targets, preds):
            pred_bands = torch.zeros_like(target)
            for class_i in np.arange(trainer.datamodule.num_classes):
                pred_bands[class_i, :, :] = pred == class_i
            pred_bands = pred_bands == 1  # draw_segmentation_masks function needs masks as bool tensors
            target = target == 1
            pred_overlay = draw_segmentation_masks(image, masks=pred_bands, colors=OCSGE_LUT, alpha=0.4)
            target_overlay = draw_segmentation_masks(image, masks=target, colors=OCSGE_LUT, alpha=0.4)
            grids.append(make_grid([image, target_overlay, pred_overlay]))
        image_grid = torch.cat(grids, 1)

        if len(self.tensorboard_logger_idx) == 1:
            pl_module.logger.experiment.add_image("Predictions", image_grid, pl_module.current_epoch)
        elif len(self.tensorboard_logger_idx) > 1:
            phase_index = self.tensorboard_logger_idx[map_phase_logger_idx(phase)]
            pl_module.logger.experiment[phase_index].add_image("Predictions", image_grid, pl_module.current_epoch)
        else:
            LOGGER.error("ERROR: the callback GraphAdder won't work if there is any Tensorboard logger.")
            raise OdeonError(ErrorCodes.ERR_CALLBACK_ERROR,
                             "GraphAdder callback is not use properly.")

    def on_train_epoch_end(self, trainer, pl_module):
        self.add_predictions(trainer=trainer, pl_module=pl_module, phase='train')

    def on_validation_epoch_end(self, trainer, pl_module):
        self.add_predictions(trainer=trainer, pl_module=pl_module, phase='val')

    def on_test_epoch_end(self, trainer, pl_module):
        self.add_predictions(trainer=trainer, pl_module=pl_module, phase='test')

    def on_fit_end(self, trainer, pl_module):
        self.train_samples = self.train_samples.detach()
        self.val_samples = self.val_samples.detach()
        return super().on_fit_end(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        self.test_samples = self.test_samples.detach()
        return super().on_test_end(trainer, pl_module)