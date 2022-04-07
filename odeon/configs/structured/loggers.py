from pydantic.dataclasses import dataclass
from typing import Any, Optional, List
from omegaconf import MISSING


@dataclass
class TensorBoardLoggerConf:
    _target_: str = "pytorch_lightning.loggers.TensorBoardLogger"
    save_dir: str = MISSING
    name: Optional[str] = "tensorboard_logs"
    version: Optional[Any] = None
    default_hp_metric: Optional[bool] =False
    sub_dir: Optional[str] = None
    filename_suffix: Optional[str] = None


@dataclass
class CSVLoggerConf:
    _target_: str = "pytorch_lightning.loggers.CSVLogger"
    save_dir: str = MISSING
    name: Optional[str] = "history_csv"
    version: Optional[Any] = None
    prefix: str = ""


@dataclass
class WandbLoggerConf:
    _target_: str = "pytorch_lightning.loggers.WandbLogger"
    name: Optional[str] = None
    save_dir: Optional[str] = None
    offline: bool = False
    id: Optional[str] = None
    anonymous: bool = False
    version: Optional[str] = None
    project: Optional[str] = None
    log_model: bool = False
    experiment: Any = None
    prefix: str = ""
