from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Union

from odeon.core.app import App
from odeon.core.types import PARAMS, STAGES_OR_VALUE, OdnLogger

# from .callbacks import CallbackRegistry, build_callbacks
from .logger import build_loggers

# from .trainer import OdnTrainer


@dataclass
class FitApp(App):

    model_name: str
    model_params: PARAMS = field(default_factory=PARAMS)
    input_name: str = 'input'
    input_params: PARAMS = field(default_factory=PARAMS)
    stages: STAGES_OR_VALUE | List[STAGES_OR_VALUE] = 'fit'
    _has_fit_stage: bool = field(init=False, default=False)
    process_position: int = 0
    num_nodes: int = 1
    devices: int = 1
    strategy: Optional[str] = None  # custom strategy not enabled yet
    accelerator: Optional[str] = None  # custom accelerator not enabled yet
    use_lr_monitor: bool = True
    lr_monitor_params: PARAMS = field(default_factory=PARAMS)
    _use_model_checkpoint: bool = field(init=False, default=False)
    model_checkpoint_params: PARAMS = field(default_factory=PARAMS)
    loggers: Dict[str, PARAMS] | OdnLogger | List[OdnLogger] = field(default_factory=Dict[str, PARAMS])
    extra_callbacks: Dict[str, PARAMS] = field(default_factory=Dict[str, PARAMS])
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: str = 'norm'
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
    profiler: Optional[str] = None

    def __post_init__(self):
        self.lr_monitor_params = {'logging_interval': 'step'} if \
            (self.lr_monitor_params is None and self.use_lr_monitor) else self.lr_monitor_params

    def run(self):
        ...

    def configure_model(self):
        ...

    def configure_input(self):
        ...

    def configure_loggers(self):
        self.loggers = build_loggers(self.loggers)

    def configure_callbacks(self):
        ...

    def configure_stages(self):
        ...

    def configure_trainer(self):
        ...
