from json import load
from sklearn.datasets import load_files
import torch
import pytorch_lightning as pl
from torchmetrics import MeanMetric

from odeon import LOGGER
from odeon.modules.metrics_module import OdeonMetrics
from odeon.nn.models import build_model
from odeon.nn.losses import build_loss_function
from odeon.nn.optim import build_optimizer, build_scheduler

PATIENCE = 30
DEFAULT_CRITERION = "ce"
DEFAULT_LR = 0.01


class SegmentationTask(pl.LightningModule):
    def __init__(self,
                 model_name,
                 num_classes,
                 num_channels,
                 class_labels,
                 criterion_name=DEFAULT_CRITERION,
                 learning_rate=DEFAULT_LR,
                 optimizer_config=None,
                 scheduler_config=None,
                 patience=PATIENCE,
                 load_pretrained_weights=None,
                 init_model_weights=None,
                 loss_classes_weights=None):

        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.class_labels = class_labels
        self.criterion_name = criterion_name
        self.learning_rate = learning_rate
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.patience = patience
        self.load_pretrained_weights = load_pretrained_weights
        self.init_model_weights = init_model_weights
        self.loss_classes_weights = self.num_classes * [1] if loss_classes_weights is None else loss_classes_weights
        print("dans l'init", self.loss_classes_weights)
        # Variables not stocked in hparams dict
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.samples = None
        self.idx_csv_loggers = None

        self.save_hyperparameters("model_name", "num_classes", "num_channels", "class_labels", "criterion_name", 
                                  "optimizer_config", "learning_rate", "scheduler_config", "patience", "load_pretrained_weights", 
                                  "init_model_weights", "loss_classes_weights")

    def setup(self, stage):
        self.model = build_model(model_name=self.hparams.model_name,
                                 n_channels=self.hparams.num_channels,
                                 n_classes=self.hparams.num_classes,
                                 init_model_weights=self.hparams.init_model_weights,
                                 load_pretrained_weights=self.hparams.load_pretrained_weights
                                 )

        self.criterion= build_loss_function(self.hparams.criterion_name, self.hparams.loss_classes_weights)

        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_epoch_metrics, self.val_epoch_metrics = None, None
            self.train_metrics = OdeonMetrics(num_classes=self.hparams.num_classes,
                                              class_labels=self.hparams.class_labels)
            self.val_metrics = OdeonMetrics(num_classes=self.hparams.num_classes,
                                            class_labels=self.hparams.class_labels)
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()

        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics = None, None
            self.val_metrics = OdeonMetrics(num_classes=self.hparams.num_classes,
                                            class_labels=self.hparams.class_labels)
            self.val_loss = MeanMetric()

        elif stage == "test":
            self.test_epoch_loss, self.test_epoch_metrics = None, None
            self.test_metrics = OdeonMetrics(num_classes=self.hparams.num_classes,
                                             class_labels=self.hparams.class_labels)
            self.test_loss = MeanMetric()

        elif stage == "predict":
            self.predict_epoch_loss, self.predict_epoch_metrics = None, None
            self.predict_metrics = OdeonMetrics(num_classes=self.hparams.num_classes,
                                                class_labels=self.hparams.class_labels)
            self.predict_loss = MeanMetric()

    def forward(self, images):
        logits = self.model(images)
        return logits

    def step(self, batch):
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
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.train_loss.update(loss)
        self.train_metrics(preds=preds, target=targets)
        return loss

    def training_epoch_end(self, outputs):
        self.train_epoch_loss = self.train_loss.compute()
        self.train_epoch_metrics = self.train_metrics.compute()
        self.log("train_loss", self.train_epoch_loss,
                    on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.val_loss.update(loss)
        self.val_metrics(preds=preds, target=targets)
        return loss

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
        loss, preds, targets = self.step(batch)
        self.test_loss.update(loss)
        self.test_metrics(preds=preds, target=targets)
        return loss

    def test_epoch_end(self, outputs):
        self.test_epoch_loss = self.test_loss.compute()
        self.test_epoch_metrics = self.test_metrics.compute()
        self.test_loss.reset()
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, targets, filenames, affines = batch["image"], batch["mask"], batch["filename"], batch["affine"]
        logits = self.model(images)

        loss = self.criterion(logits, targets)
        proba = torch.softmax(logits, dim=1)

        preds = torch.argmax(proba, dim=1)
        preds = preds.flatten(start_dim=1)

        targets = torch.argmax(targets, dim=1)
        targets = targets.flatten(start_dim=1).type(torch.int32)

        self.predict_loss.update(loss)
        self.predict_metrics(preds=preds, target=targets)

        return {"proba": proba, "filename": filenames, "affine": affines}

    def on_predict_epoch_end(self, results):
        self.predict_epoch_loss = self.predict_loss.compute()
        self.predict_epoch_metrics = self.predict_metrics.compute()
        self.predict_loss.reset()
        self.predict_metrics.reset()

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = build_optimizer(params=self.model.parameters(),
                                             learning_rate=self.hparams.learning_rate,
                                             optimizer_config=self.hparams.optimizer_config)
        if self.scheduler is None:
            self.scheduler = build_scheduler(optimizer=self.optimizer,
                                             scheduler_config=self.hparams.scheduler_config,
                                             patience=self.hparams.patience)

        config = {"optimizer": self.optimizer,
                  "lr_scheduler": self.scheduler,
                  "monitor": "val_loss"}
        return config

    def on_save_checkpoint(self, checkpoint):
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        return super().on_load_checkpoint(checkpoint)
