import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from typing import Callable, Dict
import kornia as K
from dataclasses import dataclass
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import RandomVerticalFlip, RandomRotation, RandomHorizontalFlip
from kornia.augmentation import ColorJitter, RandomResizedCrop, RandomCrop
from kornia.augmentation.augmentation import RandomGaussianBlur
from kornia.augmentation import RandomThinPlateSpline, Denormalize, Normalize
from kornia.augmentation.base import GeometricAugmentationBase2D
from torch import Tensor
from core.constants import UINT8_MAX, N_URBAIN_CLASS, IMAGENET
import torch.nn.functional as F


class SemanticSegmentationTaskAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_hard_radiometrc_transform: bool = True) -> None:

        super().__init__()
        self._apply_hard_radiometrc_transform = apply_hard_radiometrc_transform


        self.geo_transforms = nn.Sequential(
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomResizedCrop(size=(512, 512), scale=(0.4, 1.0), p=1.0)
            )

        self.hard_radiometrc_transform = ColorJitter(1.0, 1.0, 1.0, 1.0, p=1.0)
        self.soft_radiometrc_transform = RandomGaussianBlur((3, 3), (2.0, 2.0), p=1.)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:

        x_out: Dict[str, Tensor] = dict()

        img = x["img"]
        mask = x["mask"]

        if self._apply_hard_radiometrc_transform:

            img_out = self.hard_radiometrc_transform(self.geo_transforms(img))

        else:

            img_out = self.soft_radiometrc_transform(self.geo_transforms(img))

        x_out["img"] = img_out
        x_out["mask"] = mask

        return x_out


class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    def __init__(self) -> None:

        super().__init__()
        self.normalize = Normalize(mean=torch.from_numpy(np.asarray(IMAGENET["mean"])),
                                   std=torch.from_numpy(np.asarray(IMAGENET["mean"])))

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Dict[str, np.ndarray], to_squeeze: bool = True) -> Dict[str, Tensor]:

        x_out: Dict[str, Tensor] = dict()
        for key, value in x.items():

            value_out: Tensor = image_to_tensor(value)

            if "img" in key:

                value_out = value_out.float() / UINT8_MAX
                value_out = self.normalize(value_out)
                value_out = value_out.squeeze()

            else:

                value_out = value_out.float()

            x_out[key] = value_out

        return x_out


def denormalize(x,
                mean=torch.from_numpy(np.asarray(IMAGENET["mean"])),
                std=torch.from_numpy(np.asarray(IMAGENET["mean"]))):

    denorm_trans = Denormalize(mean=mean, std=std)
    return denorm_trans(x)



