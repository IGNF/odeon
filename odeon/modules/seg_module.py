import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics import Metric
from torchmetrics import MeanMetric
from omegaconf import (
    OmegaConf,
    DictConfig,
    ListConfig
)
from typing import Optional, List
from hydra.utils import instantiate
from typing import Dict
from odeon import LOGGER
from odeon.modules.metrics_module import OdeonMetrics
from odeon.configs.core import Config


def instantiate_loss(config:dict)-> torch.nn.modules.Module:
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config)
    if "weight" in config.keys() and config["weight"] is not None:
        config["weight"] = torch.Tensor(config["weight"])
    return instantiate(config)


class SegmentationTask(pl.LightningModule):
    def __init__(
        self,
        config:Config,
    )-> None: 
        super().__init__()

        self.save_hyperparameters(config)

        # Init losses
        self.train_loss: Metric = None
        self.val_loss: Metric = None
        self.test_loss: Metric = None

        # Init metrics
        self.train_metrics: OdeonMetrics = None
        self.val_metrics: OdeonMetrics = None
        self.test_metrics: OdeonMetrics = None

        # Init model
        self.model: torch.nn.Module = instantiate(self.hparams.model)
    
        # Variables not stocked in hparams dict
        self.criterion: torch.nn.modules.loss = instantiate_loss(self.hparams.loss)

        self.optimizer: torch.optim.optimizer = None
        self.scheduler: torch.optim.lr_scheduler = None
        self.samples: List[torch.Tensor] = None
        self.idx_csv_loggers: List[int] = None

    def setup(self, stage:str=None):

        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_epoch_metrics, self.val_epoch_metrics = None, None
            self.train_metrics = OdeonMetrics(num_classes=self.hparams.model.classes,
                                              class_labels=self.hparams.class_labels)
            self.val_metrics = OdeonMetrics(num_classes=self.hparams.model.classes,
                                            class_labels=self.hparams.class_labels)
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()

        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics = None, None
            self.val_metrics = OdeonMetrics(num_classes=self.hparams.model.classes,
                                            class_labels=self.hparams.class_labels)
            self.val_loss = MeanMetric()

        elif stage == "test":
            self.test_epoch_loss, self.test_epoch_metrics = None, None
            self.test_metrics = OdeonMetrics(num_classes=self.hparams.model.classes,
                                             class_labels=self.hparams.class_labels)
            self.test_loss = MeanMetric()

        elif stage == "predict":
            self.predict_epoch_loss, self.predict_epoch_metrics = None, None
            self.predict_metrics = OdeonMetrics(num_classes=self.hparams.model.classes,
                                                class_labels=self.hparams.class_labels)
            self.predict_loss = MeanMetric()

    def forward(self, images: torch.Tensor)-> torch.Tensor:
        logits = self.model(images)
        return logits

    def step(self, batch:Dict[str, torch.Tensor])-> Dict[str, torch.Tensor]:
        images, targets = batch["image"], batch["mask"]
        logits = self.forward(images)
        loss = self.criterion(logits, targets)
        with torch.no_grad():
            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(proba, dim=1)
            targets = torch.argmax(targets, dim=1)
            # Change shapes and cast target to integer for metrics computation
            preds = preds.flatten(start_dim=1)
            targets = targets.flatten(start_dim=1).type(torch.int32)
        return {"loss": loss,
                "preds": preds, 
                "targets": targets}

    def step_end(self, step_output:Dict[str, torch.Tensor], loss_meter:MeanMetric, metrics:OdeonMetrics)-> torch.Tensor:
        loss, preds, targets = step_output["loss"].mean(), step_output["preds"], step_output["targets"]
        loss_meter.update(loss)
        metrics.update(preds=preds, target=targets)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def training_step_end(self, step_output):
        return self.step_end(step_output, self.train_loss, self.train_metrics)

    def training_epoch_end(self, outputs):
        self.train_epoch_loss = self.train_loss.compute()
        self.train_epoch_metrics = self.train_metrics.compute()
        self.log("train_loss", self.train_epoch_loss,
                    on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step_end(self, step_output):
        return self.step_end(step_output, self.val_loss, self.val_metrics)

    def validation_epoch_end(self, outputs):
        self.val_epoch_loss = self.val_loss.compute()
        self.val_epoch_metrics = self.val_metrics.compute()
        # self.log: log metrics we want to monitor for model selection in checkpoints creation
        self.log("val_loss", self.val_epoch_loss,
                    on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_miou', self.val_epoch_metrics["Average/IoU"],
                    on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.step(batch)

    def test_step_end(self, step_output):
        return self.step_end(step_output, self.test_loss, self.test_metrics)

    def test_epoch_end(self, outputs):
        self.test_epoch_loss = self.test_loss.compute()
        self.test_epoch_metrics = self.test_metrics.compute()
        self.test_loss.reset()
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, filenames, affines = batch["image"], batch["filename"], batch["affine"]
        logits = self.model(images)
        proba = torch.softmax(logits, dim=1)
        return {"proba": proba,
                "filename": filenames,
                "affine": affines}

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = instantiate(self.hparams.optimizer, params=self.model.parameters())
        if self.scheduler is None:
            if self.hparams.scheduler is not None:
                self.scheduler = instantiate(self.hparams.scheduler,
                                             optimizer=self.optimizer)
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                return {'optimizer': self.optimizer,
                        'lr_scheduler': self.scheduler,
                        'monitor': 'val_loss'}
            return [self.optimizer], [self.scheduler]
        return self.optimizer

    def on_save_checkpoint(self, checkpoint):
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        return super().on_load_checkpoint(checkpoint)

