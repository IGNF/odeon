from dataclasses import field
from pydantic.dataclasses import dataclass
from typing import Any, Optional, List
from omegaconf import MISSING


# Pytorch Lightnings built-in callbacks
@dataclass
class EarlyStoppingConf:
    _target_: str = 'pytorch_lightning.callbacks.EarlyStopping'
    monitor: str = 'val_loss'
    min_delta: float = 0.0
    patience: int = 30
    verbose: bool = False
    mode: str = 'min'
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    check_on_train_epoch_end: bool = False


@dataclass
class GPUStatsMonitorConf:
    _target_: str = "pytorch_lightning.callbacks.GPUStatsMonitor"
    memory_utilization: bool = True
    gpu_utilization: bool = True
    intra_step_time: bool = False
    inter_step_time: bool = False
    fan_speed: bool = False
    temperature: bool = False


@dataclass
class GradientAccumulationSchedulerConf:
    _target_: str = "pytorch_lightning.callbacks.GradientAccumulationScheduler"
    scheduling: Any = MISSING


@dataclass
class LearningRateMonitorConf:
    _target_: str = "pytorch_lightning.callbacks.LearningRateMonitor"
    logging_interval: Optional[str] = None
    log_momentum: bool = False


@dataclass
class TimerConf:
    _target_: str = "pytorch_lightning.callbacks.Timer"
    duration: Optional[Any] = None
    interval: str = 'step'
    verbose: bool = True
    

@dataclass
class ProgressBarConf:
    _target_: str = "pytorch_lightning.callbacks.ProgressBar"
    refresh_rate: int = 1
    process_position: int = 0


@dataclass
class TQDMProgressBarConf:
    _target_: str = "pytorch_lightning.callbacks.progress.tqdm_progress.TQDMProgressBar"
    refresh_rate: int = 1


@dataclass
class ModelCheckpointConf:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: Optional[Any] = None
    filename: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: Optional[int] = None
    save_weights_only: bool = False
    mode: str = 'min'
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    every_n_val_epochs: Optional[int] = None
    period: Optional[int] = None


# Odeon Tensorboard callbacks
@dataclass
class MetricsAdderConf:
    _target_: str = "odeon.callbacks.tensorboard_callbacks.MetricsAdder"


@dataclass
class GraphAdderConf:
    _target_ = "odeon.callbacks.tensorboard_callbacks.GraphAdder"


@dataclass
class HistogramAdderConf:
    _target_: str = "odeon.callbacks.tensorboard_callbacks.HistogramAdder"


@dataclass
class PredictionsAdderConf:
    _target_: str = "odeon.callbacks.tensorboard_callbacks.PredictionsAdder"


@dataclass
class HParamsAdderConf:
    _target_: str = "odeon.callbacks.tensorboard_callbacks.HParamsAdder"


# Odeon Wandb callbacks
@dataclass
class LogConfusionMatrixConf:
    _target_: str = "odeon.callbacks.wandb_callbacks.LogConfusionMatrix"


@dataclass
class MetricsWandbConf:
    _target_: str = "odeon.callbacks.wandb_callbacks.MetricsWandb"


@dataclass
class UploadCodeAsArtifactConf:
    _target_: str = "odeon.callbacks.wandb_callbacks.UploadCodeAsArtifact"
    code_dir: Optional[str] = "wandb_defaults"
    use_git: Optional[bool] = True


# Odeon utils callbacks
@dataclass
class LightningCheckpointConf:
    _target_: str = "odeon.callbacks.utils_callbacks.LightningCheckpoint"
    dirpath: Optional[str] = None
    filename: Optional[str] = None
    monitor: Optional[str] = 'val_loss'
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: Optional[int] = 3
    save_weights_only: bool = False
    mode: str = 'min'
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    every_n_val_epochs: Optional[int] = None
    period: Optional[int] = None


@dataclass
class HistorySaverConf:
    _target_: str = "odeon.callbacks.utils_callbacks.HistorySaver"


@dataclass
class CustomPredictionWriterConf:
    _target_: str = "odeon.callbacks.utils_callbacks.CustomPredictionWriter"
    output_dir: str = MISSING
    output_type: str = MISSING
    write_interval: Optional[str] = "batch"


# Odeon legacy callbacks (continue learning, handling .pth files, predictions writer)
@dataclass
class ContinueTrainingConf:
    _target_: str = "odeon.callbacks.legacy_callbacks.ContinueTraining"
    out_dir: str = MISSING
    out_filename: str = MISSING
    save_history: Optional[bool] = False


@dataclass
class ExoticCheckPointConf:
    _target_: str = "odeon.callbacks.legacy_callbacks.ExoticCheckPoint"
    out_dir: str = MISSING
    out_filename: str = MISSING
    model_out_ext: Optional[str] = ".pth"


# Odeon callbacks for Tensorboard
@dataclass
class MetricsAdderConf:
    _target_: str = "odeon.callbacks.tensorboard_callbacks.MetricsAdder"


@dataclass
class HParamsAdderConf:
    _target_: str = "odeon.callbacks.tensorboard_callbacks.HParamsAdder"


@dataclass
class GraphAdderConf:
    _target_: str = "odeon.callbacks.tensorboard_callbacks.GraphAdder"
    samples: Optional[Any] = None


@dataclass
class HistogramAdderConf:
    _target_: str = "odeon.callbacks.tensorboard_callbacks.HistogramAdder"


@dataclass
class PredictionsAdderConf:
    _target_: str = "odeon.callbacks.tensorboard_callbacks.PredictionsAdder"
    train_samples: Optional[Any] = None
    val_samples: Optional[Any] = None
    test_samples: Optional[Any] = None
    num_predictions: Optional[int] = 3
    display_bands: Optional[List[int]] = field(default_factory=lambda: [1,2,3])
