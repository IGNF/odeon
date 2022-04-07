from pydantic.dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LightningTrainerConf:
    """Config to use for Pytorch Lightning Trainer."""
    enable_checkpointing: bool = True
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: str = 'norm'
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    devices: Any = None
    gpus: Optional[Any] = None  # Union[List[int], str, int]
    auto_select_gpus: bool = False
    tpu_cores: Optional[Any] = None  # Union[List[int], str, int]
    ipus: Optional[Any] = None
    log_gpu_memory: Optional[str] = None
    enable_progress_bar: bool = True
    overfit_batches: float = 0.0
    track_grad_norm: Any = -1  # Union[int, float, str]
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    accumulate_grad_batches: Any = 1  # Union[int, Dict[int, int], List[list]]
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: Optional[int] = -1
    min_steps: Optional[int] = None
    max_time: Optional[Any] = None  # Union[str, timedelta, Dict[str, int]]
    limit_train_batches: float = 1.0  # Union[int, float]
    limit_val_batches: float = 1.0  # Union[int, float]
    limit_test_batches: float = 1.0  # Union[int, float]
    limit_predict_batches: float = 1.0  # Union[int, float]
    val_check_interval: float = 1.0  # Union[int, float]
    log_every_n_steps: int = 50
    accelerator: Optional[Any] = None  # Union[str, Accelerator]
    strategy: Optional[str] = None
    sync_batchnorm: bool = False
    precision: int = 32
    enable_model_summary: bool = True
    weights_summary: Optional[str] = 'top'
    weights_save_path:Optional [str] = None
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Optional[Any] = None  # Union[Path, str]
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_n_epochs: Optional[int] = 0
    auto_lr_find: bool = False  # Union[bool, str]
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: bool = False  # Union[str, bool]
    plugins: Optional[Any] = None  # Union[List[Union[Plugin, ClusterEnvironment, str]], Plugin, ClusterEnvironment, str]
    amp_backend: str = 'native'
    amp_level: Optional[str] = None
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = 'max_size_cycle'
    terminate_on_nan: Optional[Any] = None
