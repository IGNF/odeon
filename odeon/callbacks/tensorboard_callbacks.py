import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, draw_segmentation_masks
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.nn.datasets import PatchDataset
from odeon.commons.metric.plots import plot_confusion_matrix

ALPHA = 0.4
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


class TensorboardCallback(pl.Callback):

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        self.tensorboard_logger_idx = self.get_tensorboard_logger_idx(trainer=trainer)

    @staticmethod
    def get_tensorboard_logger_idx(trainer):
        tensorboard_logger_idx = []
        for idx, logger in enumerate(trainer.logger.experiment):
            if isinstance(logger, torch.utils.tensorboard.writer.SummaryWriter):
                tensorboard_logger_idx.append(idx)
        return tensorboard_logger_idx

    @staticmethod
    def map_phase_logger_idx(phase):
        phase_table = {"train": 0,
                       "val": 1,
                       "test": 2}
        assert phase in phase_table.keys(), "Phases are train/val/test and predict."
        return phase_table[phase]


class MetricsAdder(TensorboardCallback):

    @rank_zero_only
    def add_metrics(self, trainer, pl_module, metric_collection, loss, phase):
        # For tensorboards loggers
        logger_idx = self.tensorboard_logger_idx[self.map_phase_logger_idx(phase)]

        trainer.logger[logger_idx].experiment.add_scalar(f"Loss",
                                                            loss,
                                                            global_step=pl_module.current_epoch)
        for key_metric in metric_collection.keys():
            if key_metric != "cm_macro" and key_metric != "cm_micro":
                trainer.logger[logger_idx].experiment.add_scalar(key_metric,
                                                                 metric_collection[key_metric],
                                                                 global_step=pl_module.current_epoch)
            elif key_metric == "cm_micro":
                fig_cm_micro = plot_confusion_matrix(metric_collection[key_metric].cpu().numpy(),
                                                     ['Positive', 'Negative'],
                                                     output_path=None,
                                                     cmap="YlGn")
                trainer.logger[logger_idx].experiment.add_figure("Metrics/ConfusionMatrix/Micro",
                                                                 fig_cm_micro,
                                                                 pl_module.current_epoch)
            elif key_metric == "cm_macro":
                fig_cm_macro = plot_confusion_matrix(metric_collection[key_metric].cpu().numpy(),
                                                     pl_module.hparams.class_labels,
                                                     output_path=None,
                                                     cmap="YlGn")
                trainer.logger[logger_idx].experiment.add_figure("Metrics/ConfusionMatrix/Macro",
                                                                 fig_cm_macro,
                                                                 pl_module.current_epoch)
                fig_cm_macro_norm = plot_confusion_matrix(metric_collection[key_metric].cpu().numpy(),
                                                          pl_module.hparams.class_labels,
                                                          output_path=None,
                                                          per_class_norm=True,
                                                          cmap="YlGn")
                trainer.logger[logger_idx].experiment.add_figure("Metrics/ConfusionMatrix/MacroNormalized",
                                                                 fig_cm_macro_norm,
                                                                 pl_module.current_epoch)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self.add_metrics(trainer=trainer,
                         pl_module=pl_module,
                         metric_collection=pl_module.train_epoch_metrics,
                         loss=pl_module.train_epoch_loss, 
                         phase='train')

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        self.add_metrics(trainer=trainer,
                         pl_module=pl_module,
                         metric_collection=pl_module.val_epoch_metrics,
                         loss=pl_module.val_epoch_loss, 
                         phase='val')

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        self.add_metrics(trainer=trainer,
                         pl_module=pl_module,
                         metric_collection=pl_module.test_epoch_metrics,
                         loss=pl_module.test_epoch_loss, 
                         phase='test')


class HParamsAdder(TensorboardCallback):

    def __init__(self):
        super().__init__()
        self.train_best_metrics, self.val_best_metrics, self.test_best_metrics, self.predict_best_metrics = None, None, None, None

    @rank_zero_only
    def add_hparams(self, trainer, pl_module, metric_dict, phase):
        hparams = {}
        for key, value in pl_module.hparams.items():
            if isinstance(value, (int, float, str, bool, torch.Tensor)):
                hparams[key] = value  # Tensorboard expect a dict and not AttributeDict()
        if len(self.tensorboard_logger_idx) == 1:
            trainer.logger.experiment.add_hparams(hparams, metric_dict)
        elif len(self.tensorboard_logger_idx) > 1:
            phase_index = self.tensorboard_logger_idx[self.map_phase_logger_idx(phase)]
            trainer.logger[phase_index].experiment.add_hparams(hparams, metric_dict)
        else:
            LOGGER.error("ERROR: the callback HParamsAdder won't work if there is any Tensorboard logger.")
            raise OdeonError(ErrorCodes.ERR_CALLBACK_ERROR,
                             "HParamsAdder callback is not use properly.")

    @rank_zero_only
    def update_best_metrics(self, input_metric_dict, used_metric_dict):
        for key in used_metric_dict.keys():
            if key != "cm_macro" and key != "cm_micro":
                if input_metric_dict[key] > used_metric_dict[key]:
                    used_metric_dict[key] = input_metric_dict[key]
        return used_metric_dict

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_best_metrics is None:
            self.train_best_metrics = {key: value for key, value in pl_module.train_epoch_metrics.items() if key != "cm_macro" and key != "cm_micro"}
        else:
            self.train_best_metrics = self.update_best_metrics(input_metric_dict=pl_module.train_epoch_metrics,
                                                               used_metric_dict=self.train_best_metrics)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_best_metrics is None:
            self.val_best_metrics = {key: value for key, value in pl_module.val_epoch_metrics.items() if key != "cm_macro" and key != "cm_micro"}
        else:
            self.val_best_metrics = self.update_best_metrics(input_metric_dict=pl_module.val_epoch_metrics,
                                                             used_metric_dict=self.val_best_metrics)

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_best_metrics is None:
            self.test_best_metrics = {key: value for key, value in pl_module.test_epoch_metrics.items() if key != "cm_macro" and key != "cm_micro"}
        else:
            self.test_best_metrics = self.update_best_metrics(input_metric_dict=pl_module.test_epoch_metrics,
                                                              used_metric_dict=self.test_best_metrics)

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        self.add_hparams(trainer, pl_module, self.train_best_metrics, 'train')
        self.add_hparams(trainer, pl_module, self.val_best_metrics, 'val')

    @rank_zero_only
    def on_exception(self, trainer, pl_module, exception):
        if self.train_best_metrics is not None:
            self.add_hparams(trainer, pl_module, self.train_best_metrics, 'train')
        if self.val_best_metrics is not None:
            self.add_hparams(trainer, pl_module, self.val_best_metrics, 'val')
        if self.test_best_metrics is not None:
            self.add_hparams(trainer, pl_module, self.test_best_metrics, 'test')
        return super().on_exception(trainer, pl_module, exception)

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        self.add_hparams(trainer, pl_module, self.test_best_metrics, 'test')


class GraphAdder(TensorboardCallback):

    def __init__(self, samples=None):
        super().__init__()
        self.samples = samples
        self.tensorboard_logger_idx = None
        self.samples = samples

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        self.tensorboard_logger_idx = self.get_tensorboard_logger_idx(trainer=trainer)
        if self.samples is None:
            self.samples = next(iter(trainer.datamodule.val_dataloader()))["image"]
        self.samples = self.samples.to(device=pl_module.device)
        if len(self.tensorboard_logger_idx) == 1:
            logger_idx = 0
            trainer.logger[logger_idx].experiment.add_graph(pl_module.model, self.samples)
        elif len(self.tensorboard_logger_idx) > 1:
            for logger_idx in self.tensorboard_logger_idx:
                trainer.logger[logger_idx].experiment.add_graph(pl_module.model, self.samples)
        else:
            LOGGER.error("ERROR: the callback GraphAdder won't work if there is any Tensorboard logger.")
            raise OdeonError(ErrorCodes.ERR_CALLBACK_ERROR,
                             "GraphAdder callback is not use properly.")
        # Detach samples from device
        self.samples = self.samples.detach()


class HistogramAdder(TensorboardCallback):

    def __init__(self):
        super().__init__()
        self.tensorboard_logger_idx = None

    @rank_zero_only
    def add_histogram(self, trainer, pl_module, phase):
        if len(self.tensorboard_logger_idx) == 1: 
            for name, params in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(name, params, pl_module.current_epoch)
        elif len(self.tensorboard_logger_idx) > 1:
            phase_index = self.tensorboard_logger_idx[self.map_phase_logger_idx(phase)]
            for name, params in pl_module.named_parameters():
                trainer.logger.experiment[phase_index].add_histogram(name, params, pl_module.current_epoch)
        else:
            LOGGER.error("ERROR: the callback HistogramAdder won't work if there is any Tensorboard logger.")
            raise OdeonError(ErrorCodes.ERR_CALLBACK_ERROR,
                             "HistogramAdder callback is not use properly.")

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self.add_histogram(trainer=trainer, pl_module=pl_module, phase='train')

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        self.add_histogram(trainer=trainer, pl_module=pl_module, phase='val')

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        self.add_histogram(trainer=trainer, pl_module=pl_module, phase='test')


class PredictionsAdder(TensorboardCallback):
    
    def __init__(
        self, train_samples=None,
        val_samples=None,
        test_samples=None,
        num_predictions=NUM_PREDICTIONS,
        display_bands=[1, 2, 3],
        ):

        super().__init__()
        self.tensorboard_logger_idx = None
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.num_predictions = num_predictions
        self.sample_dataset = None
        self.display_bands = [idx_band - 1 for idx_band in display_bands]

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        self.tensorboard_logger_idx = self.get_tensorboard_logger_idx(trainer=trainer)
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

    @rank_zero_only
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

    @rank_zero_only
    def add_predictions(self, trainer, pl_module, phase):
        if phase == "train":
            samples = self.train_samples
        elif phase == "val":
            samples = self.val_samples
        elif phase == "test" or phase =="predict":
            samples = self.test_samples

        images, targets = samples["image"].to(device=pl_module.device), samples["mask"].to(device=pl_module.device)
        with torch.no_grad():
            logits = pl_module.forward(images)
            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(proba, dim=1)

        images = images.cpu().type(torch.uint8)
        targets = targets.cpu()
        preds = preds.cpu()
        grids = []
        images = torch.stack([images[:, band_i, :, :] for band_i in self.display_bands], 1)
        for image, target, pred in zip(images, targets, preds):
            pred_bands = torch.zeros_like(target)
            for class_i in np.arange(trainer.datamodule.num_classes):
                pred_bands[class_i, :, :] = pred == class_i
            pred_bands = pred_bands == 1  # draw_segmentation_masks function needs masks as bool tensors
            target = target == 1
            pred_overlay = draw_segmentation_masks(image, masks=pred_bands, colors=OCSGE_LUT, alpha=ALPHA)
            target_overlay = draw_segmentation_masks(image, masks=target, colors=OCSGE_LUT, alpha=ALPHA)
            grids.append(make_grid([image, target_overlay, pred_overlay]))
        image_grid = torch.cat(grids, 1)

        if len(self.tensorboard_logger_idx) == 1:
            trainer.logger.experiment.add_image("Images - Masks - Predictions", image_grid, pl_module.current_epoch)
        elif len(self.tensorboard_logger_idx) > 1:
            phase_index = self.tensorboard_logger_idx[self.map_phase_logger_idx(phase)]
            trainer.logger[phase_index].experiment.add_image("Images/Masks - Predictions", image_grid, pl_module.current_epoch)
        else:
            LOGGER.error("ERROR: the callback PredictionsAdder won't work if there is any Tensorboard logger.")
            raise OdeonError(ErrorCodes.ERR_CALLBACK_ERROR,
                             "PredictionsAdder callback is not use properly.")

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self.add_predictions(trainer=trainer, pl_module=pl_module, phase='train')

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        self.add_predictions(trainer=trainer, pl_module=pl_module, phase='val')

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        self.add_predictions(trainer=trainer, pl_module=pl_module, phase='test')
