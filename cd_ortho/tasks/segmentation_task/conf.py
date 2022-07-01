import os
import datetime
from pathlib import Path
from typing import Optional, List, Union, Tuple, Any
from dataclasses import dataclass, field
from cd_ortho.core.default_conf import UrbanInputFields, NAFInputFields, EnvConf
from cd_ortho.core.constants import IMAGENET
from pytorch_lightning.utilities.seed import seed_everything


@dataclass
class TrainConf:

    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True
    batch_size: int = 6
    num_workers: int = 8


@dataclass
class ValConf(TrainConf):

    batch_size: int = 4
    shuffle: bool = False
    num_sample: int = 20
    num_workers: int = 8


@dataclass
class TestConf(ValConf):

    shuffle: bool = False
    batch_size: int = 2
    num_workers: int = 8
    num_sample: int = 20
    img_2019_field: str = None
    img_2016_style_2019_field: str = None
    img_2016_style_2016_field: str = None
    img_2019_bands: List = field(default_factory=lambda: [1, 2, 3])
    img_2016_style_2019_bands: List = field(default_factory=lambda: [6, 7, 8])
    img_2016_style_2016_bands: List = field(default_factory=lambda: [1, 2, 3])
    mask_2019_field: str = None
    mask_2016_field: str = None
    mask_change_field: str = None
    mask_2019_bands: List = field(default_factory=lambda: [1])
    mask_2016_bands: List = field(default_factory=lambda: [1])
    mask_change_bands: List = field(default_factory=lambda: [1])


@dataclass
class SegmentationTaskConf:

    project_name: str = "pocs_gers"
    nomenclature: str = "naf"
    task_name: str = "segmentation"
    data_path: str = "supervised_dataset"
    inputF: Any = None
    gpu: Union[int, List[int], List[str], str] = field(default_factory=lambda: [0])
    db_name: str = "supervised_dataset_with_stats_and_weights_2016_balanced.geojson"
    expe_name = "segmentation_mono_temporelle_mono_style_2016_training-sgd-RGB"
    img_field: str = None
    img_2_field: str = None
    multi_temporal: bool = False
    mono_temporal: str = "2016"
    mask_field: str = None
    mask_2_field: str = None
    output_size: int = 512
    is_updated_col = "2016_updated"
    img_bands: List = field(default_factory=lambda: [1, 2, 3])
    multi_style_training: bool = True
    img_other_style_bands: List = field(default_factory=lambda: [6, 7, 8])
    mask_bands: List = field(default_factory=lambda: [1])
    fold: int = 1
    model_name: str = "unet"
    encoder_name: str = "timm-efficientnet-b4"
    encoder_weights: str = "imagenet"
    pretrained: bool = True
    classes: int = None
    stage: str = "fit"
    mode: str = "soft_aug"
    mean: List = field(default_factory=lambda: IMAGENET["mean"])
    std: List = field(default_factory=lambda: IMAGENET["std"])
    imagenet_raster_band = [1, 2, 3]
    seed: int = 125
    seed_everything: bool = False
    debug: bool = False
    momentum: float = 0.9
    weight_decay: float = 1e-5
    lr: float = 1e-1
    eta_min: float = 1e-5
    t_max: int = 10
    in_chans: int = 3
    save_top_k_models: int = 5
    features_only: bool = True
    max_epochs: int = 500
    checkpoint_path: Optional[str] = None
    log_path: Optional[str] = None
    output_path: Optional[str] = None
    warmup_epochs: int = 5
    data_dir: Optional[str] = None
    path_model_output: Optional[str] = None
    path_model_checkpoint: Optional[str] = None
    path_model_log: Optional[str] = None
    path_model_examples: Optional[str] = None
    train_conf: Optional[TrainConf] = None
    val_conf: Optional[ValConf] = None
    test_conf: Optional[TestConf] = None
    env_conf: Optional[EnvConf] = None
    conf_matrix_every_n_epochs: int = 10
    print_n_batches_every_n_epochs: Tuple[int, int] = (4, 30)
    histo_weights_every_n_epochs: int = 20
    weighted_random_sampler: bool = True
    weighted_random_sampler_col: str = "weight"
    fine_tune: bool = False
    fine_tune_lr_reg: float = 0.1

    def __post_init__(self):

        self.train_conf = TrainConf() if self.train_conf is None else self.train_conf
        self.val_conf = ValConf() if self.val_conf is None else self.val_conf
        self.test_conf = TestConf() if self.test_conf is None else self.test_conf
        self.env_conf = EnvConf() if self.env_conf is None else self.env_conf
        self.expe_name = f"{self.expe_name}_run-{str(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'))}"

        if self.nomenclature == "urbain":
            self.inputF = UrbanInputFields()

        elif self.nomenclature == "naf":
            self.inputF = NAFInputFields()

        self.classes = self.inputF.classes
        if self.multi_temporal is True:

            self.img_field = self.inputF.img_1
            self.img_2_field = self.inputF.img_2
            self.mask_field = self.inputF.mask_1
            self.mask_2_field = self.inputF.mask_2

        else:

            if self.img_field is None:
                self.img_field = self.inputF.img_2 if self.mono_temporal == "2019" else self.inputF.img_1
            if self.mask_field is None:
                self.mask_field = self.inputF.mask_2 if self.mono_temporal == "2019" else self.inputF.mask_1

        self.test_conf.img_2019_field = self.inputF.img_2 if self.test_conf.img_2019_field is None else self.test_conf.img_2019_field
        self.test_conf.img_2016_style_2016_field = self.inputF.img_1 if self.test_conf.img_2016_style_2016_field is None else self.test_conf.img_2016_style_2016_field
        self.test_conf.img_2016_style_2019_field = self.inputF.img_1 if self.test_conf.img_2016_style_2019_field is None else self.test_conf.img_2016_style_2019_field
        self.test_conf.mask_2016_field = self.inputF.mask_1 if self.test_conf.mask_2016_field is None else self.test_conf.mask_2019_field
        self.test_conf.mask_2019_field = self.inputF.mask_2 if self.test_conf.mask_2019_field is None else self.test_conf.mask_2019_field
        self.test_conf.mask_change_field = self.inputF.mask_change if self.test_conf.mask_change_field is None else self.test_conf.mask_change_field
        env_conf = EnvConf()
        ROOT = env_conf.root
        self.data_dir = os.path.join(ROOT, self.data_path)
        self.path_model_output = os.path.join(env_conf.log_path, *[self.task_name, self.expe_name])
        self.path_model_checkpoint = os.path.join(self.path_model_output, "checkpoint")
        self.path_model_log = os.path.join(self.path_model_output, "logs")
        self.path_model_tests = os.path.join(self.path_model_output, "tests")
        self.path_model_predictions = os.path.join(self.path_model_output, "predictions")
        self.path_model_examples = os.path.join(self.path_model_output, "examples")

        if os.path.isdir(self.path_model_output) is False:
            Path(self.path_model_output).mkdir(parents=True)
        if os.path.isdir(self.path_model_checkpoint) is False:
            Path(self.path_model_checkpoint).mkdir(exist_ok=True)
        if os.path.isdir(self.path_model_log) is False:
            Path(self.path_model_log).mkdir(exist_ok=True)
        if os.path.isdir(self.path_model_examples) is False:
            Path(self.path_model_examples).mkdir(exist_ok=True)
        if os.path.isdir(self.path_model_tests) is False:
            Path(self.path_model_tests).mkdir(exist_ok=True)
        if os.path.isdir(self.path_model_predictions) is False:
            Path(self.path_model_predictions).mkdir(exist_ok=True)

        if self.seed_everything:
            seed_everything(self.seed, workers=True)
        if self.weighted_random_sampler:
            self.weighted_random_sampler_col = f"{self.weighted_random_sampler_col}_{self.nomenclature}"

        assert len(self.img_bands) == len(self.img_other_style_bands)
        assert len(self.img_bands) == len(self.test_conf.img_2019_bands) == len(self.test_conf.img_2016_style_2016_bands)
        assert len(self.test_conf.img_2019_bands) == len(self.img_bands)

        if len(self.img_bands) != 3:

            assert len(self.img_bands) == len(self.imagenet_raster_band)
            self.mean = [IMAGENET["mean"][i - 1] for i in self.imagenet_raster_band]
            self.std = [IMAGENET["std"][i - 1] for i in self.imagenet_raster_band]
