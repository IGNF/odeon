from pydantic.dataclasses import dataclass
from typing import Any
from omegaconf import MISSING


@dataclass
class ReduceLROnPlateauConf:
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: Any = "min"
    factor: Any = 0.1
    patience: Any = 10
    threshold: Any = 0.0001
    threshold_mode: Any = "rel"
    cooldown: Any = 0
    min_lr: Any = 0
    eps: Any = 1e-08
    verbose: Any = False


@dataclass
class CosineAnnealingLRConf:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    T_max: Any = MISSING
    eta_min: Any = 0
    last_epoch: Any = -1
    verbose: Any = False


@dataclass
class CosineAnnealingWarmRestartsConf:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
    T_0: Any = MISSING
    T_mult: Any = 1
    eta_min: Any = 0
    last_epoch: Any = -1
    verbose: Any = False


@dataclass
class CyclicLRConf:
    _target_: str = "torch.optim.lr_scheduler.CyclicLR"
    base_lr: Any = MISSING
    max_lr: Any = MISSING
    step_size_up: Any = 2000
    step_size_down: Any = None
    mode: Any = "triangular"
    gamma: Any = 1.0
    scale_fn: Any = None
    scale_mode: Any = "cycle"
    cycle_momentum: Any = True
    base_momentum: Any = 0.8
    max_momentum: Any = 0.9
    last_epoch: Any = -1
    verbose: Any = False


@dataclass
class ExponentialLRConf:
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    gamma: Any = MISSING
    last_epoch: Any = -1
    verbose: Any = False


@dataclass
class LambdaLRConf:
    _target_: str = "torch.optim.lr_scheduler.LambdaLR"
    lr_lambda: Any = MISSING
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class MultiplicativeLRConf:
    _target_: str = "torch.optim.lr_scheduler.MultiplicativeLR"
    lr_lambda: Any = MISSING
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class MultiStepLRConf:
    _target_: str = "torch.optim.lr_scheduler.MultiStepLR"
    milestones: Any = MISSING
    gamma: Any = 0.1
    last_epoch: Any = -1
    verbose: Any = False


@dataclass
class OneCycleLRConf:
    _target_: str = "torch.optim.lr_scheduler.OneCycleLR"
    max_lr: Any = MISSING
    total_steps: Any = None
    epochs: Any = None
    steps_per_epoch: Any = None
    pct_start: Any = 0.3
    anneal_strategy: Any = "cos"
    cycle_momentum: Any = True
    base_momentum: Any = 0.85
    max_momentum: Any = 0.95
    div_factor: Any = 25.0
    final_div_factor: Any = 10000.0
    last_epoch: Any = -1
    verbose: Any = False


@dataclass
class StepLRConf:
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: Any = MISSING
    gamma: Any = 0.1
    last_epoch: Any = -1
    verbose: Any = False
