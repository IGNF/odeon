from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

from odeon.core.app_runner import AbsAppRunner

# from .trainer import OdnTrainer

PARAMS = Dict[str, Any]


@dataclass
class FitRunner(AbsAppRunner):

    model_name: str
    model_params: PARAMS = field(default_factory=lambda x: dict())
    input_name: str = 'input'
    input_params: PARAMS = field(default_factory=lambda x: dict())
    use_lr_monitor: bool = True
    lr_monitor_params: PARAMS = field(default_factory=lambda x: {'logging_interval': 'step'})
    _use_model_checkpoint: bool = field(init=False, default=False)
    model_checkpoint_params: PARAMS = field(default_factory=lambda x: dict())
    logger_name: str = 'tensorboard'
    logger_params: PARAMS = field(default_factory=lambda x: dict())
    extra_callbacks: Dict[str, PARAMS] = field(default_factory=lambda x: dict())
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: str = 'norm'
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Optional[Union[List[int], str, int]] = None
    auto_select_gpus: bool = False
    tpu_cores: Optional[Union[List[int], str, int]] = None
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: Optional[int] = None
    overfit_batches: Union[int, float] = 0.0
    track_grad_norm: Union[int, float, str] = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: Union[int, bool] = False
    accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0
    limit_predict_batches: Union[int, float] = 1.0
    val_check_interval: Union[int, float] = 1.0
    flush_logs_every_n_steps: int = 100
    log_every_n_steps: int = 50
