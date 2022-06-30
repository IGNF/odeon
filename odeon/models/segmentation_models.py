import segmentation_models_pytorch as smp
from typing import Optional
import torch
from segmentation_models_pytorch.base import SegmentationModel


class SegmentationModelFactory:

    def __init__(self,
                 model_name: str = "unet",
                 encoder_name: str = "resnet18",
                 pretrained: bool = True,
                 in_chans: int = 3,
                 classes: int = 16,
                 checkpoint: Optional[str] = None,
                 features_only: bool = False):

        self.model_name: str = model_name
        self.encoder_name: str = encoder_name
        self.pretrained: bool = pretrained
        self.in_chans: int = in_chans
        self.classes = classes
        self.checkpoint: str =checkpoint
        self.features_only: bool = features_only

    def get(self) -> torch.nn.Module:

        model: smp.base.model = smp.create_model(arch=self.model_name,
                                                  encoder_name=self.encoder_name,
                                                  classes=self.classes,
                                                  in_channels=self.in_chans,
                                                  encoder_weights="imagenet")

        return model
