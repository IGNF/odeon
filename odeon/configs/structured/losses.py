from pydantic.dataclasses import dataclass
from typing import Any, Optional, List
from omegaconf import MISSING


@dataclass
class BCEWithLogitsLossConf: 
    _target_: str = "odeon.nn.losses.BCEWithLogitsLoss"
    weight: Optional[List] = None
    reduction: str = "mean"
    pos_weight: Optional[List] = None


@dataclass
class CrossEntropyWithLogitsLossConf:
    _target_: str = "odeon.nn.losses.CrossEntropyWithLogitsLoss"
    weight: Optional[List] = None
    reduction: str = "mean"


@dataclass
class ComboLossConf:
    _target_: str = "odeon.nn.losses.ComboLoss"
    weight: Optional[List] = None
    per_image: Optional[bool] = False


@dataclass
class SoftDiceLossConf:
    _target_: str = "odeon.nn.losses.SoftDiceLoss"
    weight: Optional[List] = None
    size_average: Optional[bool] = False


@dataclass
class DiceLossConf:
    _target_: str = "odeon.nn.losses.DiceLoss"
    weight: Optional[List] = None
    size_average: Optional[bool] = False
    per_image: Optional[bool] = False


@dataclass
class JaccardLossConf:
    _target_: str = "odeon.nn.losses.JaccardLoss"
    weight: Optional[List] = None
    size_average: Optional[bool] = False
    per_image: Optional[bool] = False
    non_empty: Optional[bool] = False
    apply_sigmoid: Optional[bool] = False
    min_pixels: Optional[int] = 5


@dataclass
class LovaszLossConf:
    _target_: str = "odeon.nn.losses.LovaszLoss"
    ignore_index: Optional[int] = None
    per_image: Optional[bool] = False


@dataclass
class FocalLoss2dConf:
    _target_: str = "odeon.nn.losses.FocalLoss2d"
    gamma: Optional[float] = 2.0
    ignore_index: Optional[int] = None
