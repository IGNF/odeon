# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Foreground-Aware Relation Network (FarSeg) implementations."""

import math
from collections import OrderedDict
from typing import List, cast
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import (
    BatchNorm2d,
    Conv2d,
    Identity,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Sigmoid,
    UpsamplingBilinear2d
)
from torchvision.models import resnet
from torchvision.ops import FeaturePyramidNetwork as FPN

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.models"
ModuleList.__module__ = "models.ModuleList"
Sequential.__module__ = "models.Sequential"
Conv2d.__module__ = "models.Conv2d"
BatchNorm2d.__module__ = "models.BatchNorm2d"
ReLU.__module__ = "models.ReLU"
UpsamplingBilinear2d.__module__ = "models.UpsamplingBilinear2d"
Sigmoid.__module__ = "models.Sigmoid"
Identity.__module__ = "models.Identity"


class _FSRelation(Module):
    """F-S Relation module."""

    def __init__(
        self,
        scene_embedding_channels: int,
        in_channels_list: List[int],
        out_channels: int,
    ) -> None:
        """Initialize the _FSRelation module.
        Args:
            scene_embedding_channels: number of scene embedding channels
            in_channels_list: a list of input channels
            out_channels: number of output channels
        """
        super().__init__()

        self.scene_encoder = ModuleList(
            [
                Sequential(
                    Conv2d(scene_embedding_channels, out_channels, 1),
                    ReLU(True),
                    Conv2d(out_channels, out_channels, 1),
                )
                for _ in range(len(in_channels_list))
            ]
        )

        self.content_encoders = ModuleList()
        self.feature_reencoders = ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(
                Sequential(
                    Conv2d(c, out_channels, 1),
                    BatchNorm2d(out_channels),  # type: ignore[no-untyped-call]
                    ReLU(True),
                )
            )
            self.feature_reencoders.append(
                Sequential(
                    Conv2d(c, out_channels, 1),
                    BatchNorm2d(out_channels),  # type: ignore[no-untyped-call]
                    ReLU(True),
                )
            )

        self.normalizer = Sigmoid()

    def forward(self, scene_feature: Tensor, features: List[Tensor]) -> List[Tensor]:
        """Forward pass of the model."""
        # [N, C, H, W]
        content_feats = [
            c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)
        ]
        scene_feats = [op(scene_feature) for op in self.scene_encoder]
        relations = [
            self.normalizer((sf * cf).sum(dim=1, keepdim=True))
            for sf, cf in zip(scene_feats, content_feats)
        ]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats
