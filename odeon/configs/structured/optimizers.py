from pydantic.dataclasses import dataclass
from typing import Any
from omegaconf import MISSING


@dataclass
class SGDConf:
    _target_: str = "torch.optim.SGD"
    lr: Any = None  # _RequiredParameter
    momentum: Any = 0
    dampening: Any = 0
    weight_decay: Any = 0
    nesterov: Any = False


@dataclass
class AdadeltaConf:
    _target_: str = "torch.optim.Adadelta"
    lr: Any = None
    rho: Any = 0.9
    eps: Any = 1e-06
    weight_decay: Any = 0


@dataclass
class AdagradConf:
    _target_: str = "torch.optim.Adagrad"
    lr: Any = None
    lr_decay: Any = 0
    weight_decay: Any = 0
    initial_accumulator_value: Any = 0
    eps: Any = 1e-10


@dataclass
class AdamConf:
    _target_: str = "torch.optim.Adam"
    lr: Any = None
    betas: Any = (0.9, 0.999)
    eps: Any = 1e-08
    weight_decay: Any = 0
    amsgrad: Any = False


@dataclass
class AdamaxConf:
    _target_: str = "torch.optim.Adamax"
    lr: Any = None
    betas: Any = (0.9, 0.999)
    eps: Any = 1e-08
    weight_decay: Any = 0


@dataclass
class AdamWConf:
    _target_: str = "torch.optim.AdamW"
    lr: Any = None
    betas: Any = (0.9, 0.999)
    eps: Any = 1e-08
    weight_decay: Any = 0.01
    amsgrad: Any = False


@dataclass
class ASGDConf:
    _target_: str = "torch.optim.ASGD"
    lr: Any = None
    lambd: Any = 0.0001
    alpha: Any = 0.75
    t0: Any = 1000000.0
    weight_decay: Any = 0


@dataclass
class LBFGSConf:
    _target_: str = "torch.optim.LBFGS"
    lr: Any = None
    max_iter: Any = 20
    max_eval: Any = None
    tolerance_grad: Any = 1e-07
    tolerance_change: Any = 1e-09
    history_size: Any = 100
    line_search_fn: Any = None


@dataclass
class RMSpropConf:
    _target_: str = "torch.optim.RMSprop"
    lr: Any = None
    alpha: Any = 0.99
    eps: Any = 1e-08
    weight_decay: Any = 0
    momentum: Any = 0
    centered: Any = False


@dataclass
class RpropConf:
    _target_: str = "torch.optim.Rprop"
    lr: Any = None
    etas: Any = (0.5, 1.2)
    step_sizes: Any = (1e-06, 50)


@dataclass
class SparseAdamConf:
    _target_: str = "torch.optim.SparseAdam"
    lr: Any = None
    betas: Any = (0.9, 0.999)
    eps: Any = 1e-08
