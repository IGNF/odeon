from typing import Optional, cast

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from segmentation_models_pytorch import create_model

# from odeon.models.core.models import MODEL_REGISTRY

__all__ = ['SegmentationModule']


class SegmentationModule(LightningModule):
    """Segmentation module for multiclass segmentation with pytorch segmentation models

    """
    def __init__(self,
                 classes: int,
                 model_name: str,
                 loss: nn.Module = cast(nn.Module, nn.CrossEntropyLoss),
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = None,
                 in_channels: int = 3,
                 model: Optional[nn.Module] = None,
                 **kwargs
                 ):

        super(SegmentationModule, self).__init__()

        self.model = create_model(arch=model_name,
                                  encoder_name=encoder_name,
                                  encoder_weights=encoder_weights,
                                  in_channels=in_channels,
                                  classes=classes,
                                  **kwargs) if model is None else model

        self.criterion = loss(reduction='mean')

    def training_step(self, sample, *args, **kwargs) -> STEP_OUTPUT:

        logit = self.model(sample["input"])
        pred = torch.softmax(logit, dim=1)
        # backpropagation
        loss = self.criterion(pred, sample["target"].squeeze())
        return {"loss": loss}

    def validation_step(self, sample, *args, **kwargs) -> Optional[STEP_OUTPUT]:

        logit = self.model(sample["input"])
        pred = torch.softmax(logit, dim=1)
        # backpropagation
        loss = self.criterion(pred, sample["target"].squeeze())
        return {"loss": loss}

    def test_step(self, sample, *args, **kwargs) -> Optional[STEP_OUTPUT]:

        logit = self.model(sample["input"])
        pred = torch.softmax(logit, dim=1)
        return {"pred": pred}
