import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from odeon.nn.losses import (
    BCEWithLogitsLoss,
    CrossEntropyWithLogitsLoss,
    FocalLoss2d,
    ComboLoss
)
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from odeon import LOGGER
from odeon.commons.metric.plots import plot_confusion_matrix, makegrid
from odeon.modules.metrics_module import OdeonMetrics

PATIENCE = 30


class SegmentationTask(pl.LightningModule):
    def __init__(self,
                 model,
                 num_classes,
                 class_labels,
                 criterion,
                 optimizer,
                 learning_rate,
                 scheduler=None,
                 patience=PATIENCE,
                 log_histogram=False,
                 log_graph=False,
                 log_predictions=False,
                 weights=None):

        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.patience = patience
        self.log_histogram = log_histogram
        self.log_graph = log_graph
        self.log_predictions = log_predictions
        self.samples = None
        self.weights = weights
        self.idx_csv_loggers = None

        if isinstance(criterion, str):
            self.criterion= self.get_loss_function(criterion, self.weights)
        else:
            self.criterion = criterion

    def setup(self, stage):
        if stage == "fit":
            self.train_metrics = OdeonMetrics(num_classes=self.num_classes,
                                              class_labels=self.class_labels)
            self.val_metrics = OdeonMetrics(num_classes=self.num_classes,
                                            class_labels=self.class_labels)
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()
        if stage == "test":
            self.test_metrics = OdeonMetrics(num_classes=self.num_classes,
                                             class_labels=self.class_labels)
            self.test_loss = MeanMetric()

        if self.logger is not None:
            self.idx_csv_loggers = [idx for idx, logger in enumerate(self.logger.experiment)\
                if isinstance(logger, pl.loggers.csv_logs.ExperimentWriter)]    


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

    def training_step_end(self, outputs):
        pass

    def training_epoch_end(self, outputs):
        train_epoch_loss = self.train_loss.compute()
        train_epoch_metrics = self.train_metrics.compute()
        self.logging(train_epoch_metrics, train_epoch_loss, phase='train')
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.val_loss.update(loss)
        self.val_metrics(preds=preds, target=targets)
        return loss

    def validation_step_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        val_epoch_loss = self.val_loss.compute()
        val_epoch_metrics = self.val_metrics.compute()
        self.logging(val_epoch_metrics, val_epoch_loss, phase='val') # Pass metric collection value to the tensorboard
        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.test_loss.update(loss)
        self.test_metrics(preds=preds, target=targets)
        return loss

    def test_step_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        test_epoch_loss = self.test_loss.compute()
        test_epoch_metrics = self.test_metrics.compute()
        self.logging(test_epoch_metrics, test_epoch_loss, phase='test')
        self.test_loss.reset()
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, targets = batch["image"], batch["mask"]
        logits = self.model(images)
        proba = torch.softmax(logits, dim=1)
        preds = torch.argmax(proba, dim=1)
        return {"proba": proba, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        if isinstance(self.optimizer, str):
            if self.optimizer == 'adam':
                optimizer= optim.Adam(self.model.parameters(), lr=self.learning_rate)
            elif self.optimizer == 'SGD':
                optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)    

        if self.scheduler is False:
            return optimizer
        elif self.scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer,
                                        'min',
                                        factor=0.5,
                                        patience=self.patience,
                                        cooldown=4,
                                        min_lr=1e-7)
        else:
            scheduler = self.scheduler

        config = {"optimizer": optimizer,
                  "lr_scheduler": scheduler,
                  "monitor": "val_loss"}
        return config

    def get_logger_index(self, phase):
        dict_logger_index = {'train': 0, 'val' : 1, 'test': 2}
        return dict_logger_index[phase]

    def logging(self, metric_collection, loss, phase):
        logger_idx = self.get_logger_index(phase)

        # For tensorboards loggers
        if self.logger is not None:
            self.logger.experiment[logger_idx].add_scalar(f"Loss",
                                                           loss,
                                                           global_step=self.current_epoch)
            for key_metric in metric_collection.keys():
                if key_metric != "cm_macro" and key_metric != "cm_micro":
                    self.logger.experiment[logger_idx].add_scalar(key_metric,
                                                                  metric_collection[key_metric],
                                                                  global_step=self.current_epoch)
                elif key_metric == "cm_micro":
                    fig_cm_micro = plot_confusion_matrix(metric_collection[key_metric],
                                                         ['Positive', 'Negative'],
                                                         output_path=None,
                                                         cmap="YlGn")
                    self.logger.experiment[logger_idx].add_figure("Confusion Matrix/Micro",
                                                                  fig_cm_micro,
                                                                  self.current_epoch)   
                elif key_metric == "cm_macro":
                    fig_cm_macro = plot_confusion_matrix(metric_collection[key_metric],
                                                         self.class_labels,
                                                         output_path=None,
                                                         cmap="YlGn")
                    self.logger.experiment[logger_idx].add_figure("Confusion Matrix/Macro",
                                                                 fig_cm_macro,
                                                                 self.current_epoch)
                    fig_cm_macro_norm = plot_confusion_matrix(metric_collection[key_metric],
                                                              self.class_labels,
                                                              output_path=None,
                                                              per_class_norm=True,
                                                              cmap="YlGn")
                    self.logger.experiment[logger_idx].add_figure("Confusion Matrix/Macro Normalized",
                                                                 fig_cm_macro_norm,
                                                                 self.current_epoch)

        # Log metrics for checkpoints saving and write logs in a csv file
        if phase == "train":
                    self.log("train_loss", loss,
                             on_step=False, on_epoch=True, prog_bar=True, logger=False)
        elif phase == "val":
            # self.log: log metrics we want to monitor for model selection in checkpoints creation
            self.log("val_loss", loss,
                     on_step=False, on_epoch=True, prog_bar=True, logger=False)
            self.log('val_miou', metric_collection["Average/IoU"],
                     on_step=False, on_epoch=True, prog_bar=True, logger=False)
            # Write log in a csv file
            if self.idx_csv_loggers:
                for logger_idx in self.idx_csv_loggers:
                    metric_collection['learning rate'] = self.learning_rate  # Add learning rate logging  
                    self.logger.experiment[logger_idx].log_metrics(metric_collection, self.current_epoch)
                    self.logger.experiment[logger_idx].save()

    def get_loss_function(self, loss_name, class_weight=None):
        if loss_name == "ce":
            if class_weight is not None:
                LOGGER.info(f"Weights used: {class_weight}")
                class_weight = torch.FloatTensor(class_weight)
            return CrossEntropyWithLogitsLoss(weight=class_weight)
        elif loss_name == "bce":
            return BCEWithLogitsLoss()
        elif loss_name == "focal":
            return FocalLoss2d()
        elif loss_name == "combo":
            return ComboLoss({'bce': 0.75, 'jaccard': 0.25})
