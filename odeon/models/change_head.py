# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ChangeStar implementations."""

from typing import Dict, List
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.modules import Module


# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.models"


class ChangeMixin(Module):
    """This module enables any segmentation model to detect binary change.
    The common usage is to attach this module on a segmentation model without the
    classification head.
    If you use this model in your research, please cite the following paper:
    * https://arxiv.org/abs/2108.07002
    """

    def __init__(
        self,
        in_channels: int = 128 * 2,
        inner_channels: int = 16,
        num_convs: int = 4,
        scale_factor: float = 4.0,
    ):
        """Initializes a new ChangeMixin module.
        Args:
            in_channels: sum of channels of bitemporal feature maps
            inner_channels: number of channels of inner feature maps
            num_convs: number of convolution blocks
            scale_factor: number of upsampling factor
        """
        super().__init__()
        layers: List[Module] = [
            nn.modules.Sequential(
                nn.modules.Conv2d(in_channels, inner_channels, 3, 1, 1),
                nn.modules.BatchNorm2d(inner_channels),  # type: ignore[no-untyped-call]
                nn.modules.ReLU(True),
            )
        ]
        layers += [
            nn.modules.Sequential(
                nn.modules.Conv2d(inner_channels, inner_channels, 3, 1, 1),
                nn.modules.BatchNorm2d(inner_channels),  # type: ignore[no-untyped-call]
                nn.modules.ReLU(True),
            )
            for _ in range(num_convs - 1)
        ]

        cls_layer = nn.modules.Conv2d(inner_channels, 1, 3, 1, 1)

        layers.append(cls_layer)
        layers.append(nn.modules.UpsamplingBilinear2d(scale_factor=scale_factor))

        self.convs = nn.modules.Sequential(*layers)

    def forward(self, bi_feature: Tensor) -> List[Tensor]:
        """Forward pass of the model.
        Args:
            bi_feature: input bitemporal feature maps of shape [b, t, c, h, w]
        Returns:
            a list of bidirected output predictions
        """
        batch_size = bi_feature.size(0)
        t1t2 = torch.cat(  # type: ignore[attr-defined]
            [bi_feature[:, 0, :, :, :], bi_feature[:, 1, :, :, :]], dim=1
        )
        t2t1 = torch.cat(  # type: ignore[attr-defined]
            [bi_feature[:, 1, :, :, :], bi_feature[:, 0, :, :, :]], dim=1
        )

        c1221 = self.convs(torch.cat([t1t2, t2t1], dim=0))  # type: ignore[attr-defined]
        c12, c21 = torch.split(
            c1221, batch_size, dim=0
        )  # type: ignore[no-untyped-call]
        return [c12, c21]
