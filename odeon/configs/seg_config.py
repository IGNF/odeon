from dataclasses import dataclass, fields

from typing import (
    Dict,
    Tuple,
    Optional,
    Union,
    List
)

@dataclass
class Files:
    output_folder: str
    name_exp_log: str
    version_name: str
    model_filename: str
    model_out_ext: Optional[str]


@dataclass
class DataModule:
    # Files pointing to the data
    train_file: str
    val_file: str
    test_file: Optional[str]

    # Data features
    resolution: Optional[Union[float, Tuple[float]]]

    # Dataset definition
    image_bands: Optional[List[int]]
    mask_bands: Optional[List[int]]
    width: Optional[int]
    height: Optional[int]
    data_augmentation: Optional[Dict[str]]
    get_sample_info: Optional[bool]
    percentage_val: Optional[float]
    deterministic: Optional[bool]

    # Dataloaders definition
    batch_size: Union[int, List[int]]
    num_workers: Optional[int]
    pin_memory: Optional[bool]
    subset: Optional[bool]


@dataclass
class Model:
    model_name: str
    num_classes: int
    num_channels: int
    criterion_name: str
    learning_rate: float
    log_learning_rate: Optional[bool]
    class_labels: Optional[List[str]]
    optimizer_config: Optional[dict]
    scheduler_config: Optional[dict]
    patience: Optional[int]
    load_pretrained_weights: Optional[bool]
    init_model_weights: Optional[bool]
    loss_classes_weights: Optional[List[float]]


@dataclass
class Prediction:
    resolution: Optional[Union[float, List[float]]]
    prediction_output_type: Optional[str]


@dataclass
class Loggers:
    output_tensorboard_logs: Optional[str]
    save_history: Optional[bool]
    get_prediction: Optional[bool]
    save_top_k: Optional[int]
    log_histogram: Optional[bool]
    log_graph: Optional[bool]
    log_predictions: Optional[bool]
    log_learning_rate: Optional[bool]
    log_hparams: Optional[bool]
    use_wandb: Optional[bool]


@dataclass
class Callbacks:
    early_stopping: Optional[Union[bool, str]]
    progress: Optional[Union[int, float]]


@dataclass
class Device:
    # Define device for training
    device: str
    accelerator: str
    num_nodes: Optional[int]
    num_processes: Optional[int]
    strategy: Optional[str]


@dataclass
class Params:
    # Parameters of training
    epochs: int
    batch_size: int
    val_check_interval: Optional[Union[int, float]]
    reproducible: Optional[bool]


@dataclass
class Trainer:
    device: Device
    params: Params
    loggers: Loggers
    callbacks: Callbacks


@dataclass
class OCSGEConfig:
    files: Files
    datamodule: DataModule
    model: Model
    trainer: Trainer
