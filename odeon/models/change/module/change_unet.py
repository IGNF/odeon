"""Segmentation tasks."""

# import warnings
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, cast

# import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor
# from torch.nn.functional import one_hot
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
# from torchmetrics import Metric
from torchmetrics import MetricCollection
from torchmetrics.classification import (  # type: ignore[attr-defined]
    BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex, BinaryPrecision,
    BinaryRecall, BinarySpecificity)

from odeon.core.types import OdnMetric
from odeon.models.change.arch.change_unet import FCSiamConc, FCSiamDiff
from odeon.models.core.models import ModelRegistry

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"  # Sphinx bug


@ModelRegistry.register_class(name='change_unet', aliases=['c_unet'])
class ChangeUnet(pl.LightningModule):
    """

    """
    def __init__(self,
                 model: str = 'fc_siam_conc',
                 model_params: Optional[Dict] = None,
                 loss: str = 'bce',
                 lr: float = 0.0001,
                 threshold: float = 0.5,
                 **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()
        self.model = self.configure_model(model=model, model_params=model_params)
        self.loss = self.configure_loss(loss=loss)
        self.train_metrics, self.val_metrics, self.test_metrics = self.configure_metrics(metric_params={})
        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)
        self.lr = lr
        self.activation: Callable[[Tensor], Tensor] = torch.sigmoid
        self.threshold: float = threshold
        """"
        if not isinstance(kwargs["ignore_index"], (int, type(None))):
            raise ValueError("ignore_index must be an int or None")
        if (kwargs["ignore_index"] is not None) and (kwargs["loss"] == "jaccard"):
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )
        self.ignore_index = kwargs["ignore_index"]
        """

    def configure_model(self,
                        model: str = 'fc_siam_conc',
                        model_params: Optional[Dict] = None) -> nn.Module:
        """
        Configures the task based on kwargs parameters passed to the constructor.
        Parameters
        ----------
        model
        model_params

        Returns
        -------

        """

        if model_params is None:
            model_params = {}
        if model == "fc_siam_diff":
            return FCSiamDiff(**model_params)
        elif model == "fc_siam_conc":
            return FCSiamConc(**model_params)
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                f"Currently, only supports 'unet'."
            )

    def configure_loss(self,
                       loss: str) -> nn.Module:
        if loss == "bce":
            # ignore_value = -1000 if self.ignore_index is None else self.ignore_index
            return nn.BCEWithLogitsLoss(reduction='mean')
        elif loss == "focal":
            return smp.losses.FocalLoss("binary", normalized=True)
        else:
            raise ValueError(f"Loss type '{loss}' is not valid. "
                             f"Currently, supports 'bce', or 'focal' loss.")

    def configure_lr(self,
                     lr: float,
                     optimizer: str = 'sgd',
                     scheduler: str = '',
                     differential: Optional[Dict[str, float]] = None):
        ...

    def configure_metrics(self, metric_params: Dict) -> Tuple[OdnMetric, OdnMetric, OdnMetric]:

        train_metrics = MetricCollection(
            {"bin_acc": BinaryAccuracy(),
             "bin_iou": BinaryJaccardIndex(),
             "bin_rec": BinaryRecall(),
             "bin_spec": BinarySpecificity(),
             "bin_pre": BinaryPrecision(),
             "bin_f1": BinaryF1Score()},
            prefix="train_")
        val_metrics = train_metrics.clone(prefix="val_")
        test_metrics = train_metrics.clone(prefix="test_")
        return train_metrics, val_metrics, test_metrics

    def forward(self, T0: Tensor, T1: Tensor, *args: Any, **kwargs: Any) -> Any:
        """

        Parameters
        ----------
        T0
        T1
        args
        kwargs

        Returns
        -------

        """
        x = torch.stack(tensors=(T0, T1), dim=1)
        return self.model(x)

    def configure_activation(self, activation: str, dim=1) -> Callable[[Tensor], Tensor]:
        match activation:
            case 'softmax':
                return partial(torch.softmax(dim=1))
            case 'sigmoid':
                return torch.sigmoid
            case _:
                raise RuntimeError('something went in configuration activation')

    def step(self, batch: Dict) -> Any:
        T0 = batch['T0']
        T1 = batch['T1']
        y = batch['mask']
        y_hat = self(T0=T0, T1=T1)

        return y_hat, y

    def training_step(self, batch: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """

        Parameters
        ----------
        batch
        args
        kwargs

        Returns
        -------

        """
        y_hat, y = self.step(batch=batch)
        y_hat_hard = y_hat > self.threshold
        loss = self.loss(y_hat, y.float())
        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)
        return {'loss': loss}

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Parameters
        ----------
        outputs: list of items returned by training_step

        Returns
        -------

        """

        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """

        Parameters
        ----------
        batch
        args
        kwargs

        Returns
        -------

        """
        print(batch)
        y_hat, y = self.step(batch=batch)
        y_hat_hard = y_hat > self.threshold
        loss = self.loss(y_hat, y.float())
        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)
        """
                if batch_idx < 10:
                    try:
                        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
                        batch["prediction"] = y_hat_hard
                        for key in ["image", "mask", "prediction"]:
                            batch[key] = batch[key].cpu()
                        sample = batch[0]
                        fig = datamodule.plot(sample)
                        summary_writer = self.logger.experiment  # type: ignore[union-attr]
                        summary_writer.add_figure(
                            f"image/{batch_idx}", fig, global_step=self.global_step
                        )
                        plt.close()
                    except AttributeError:
                        pass
        """
        return {'val_loss': cast(Tensor, loss)}

    def validation_epoch_end(self, outputs: Any) -> None:
        """

        Parameters
        ----------
        outputs

        Returns
        -------

        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, batch: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """

        Parameters
        ----------
        batch
        args
        kwargs

        Returns
        -------

        """
        y_hat, y = self.step(batch=batch)
        y_hat_hard = y_hat > self.threshold
        loss = self.loss(y_hat, y)
        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)
        return {'test_loss': cast(Tensor, loss)}

    def test_epoch_end(self, outputs: Any) -> None:
        """
        Parameters
        ----------
        outputs

        Returns
        -------

        """
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.
        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=20,
                ),
                "monitor": "val_loss",
            },
        }
