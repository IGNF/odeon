from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data import DataLoader, Dataset

from .dataframe import create_dataframe_from_file, split_dataframe
from .dataset import UniversalDataset
from .runner_utils import Stages
from .transform import AlbuTransform
from .types import DATAFRAME, STAGES, Overlap

DEFAULT_DATALOADER_OPTIONS = {"batch_size": 8, "num_workers": 1}


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class Input(LightningDataModule):

    input_fields: Dict
    input_fit_file: Optional[str] = None
    input_validate_file: Optional[str] = None
    input_test_file: Optional[str] = None
    input_predict_file: Optional[str] = None
    root_dir: Union[None, str, Dict] = None
    input_files_has_header: Union[bool, Dict] = True  # Rather input files have header or not
    fit_transform: Union[List[Callable], None] = None
    validate_transform: Union[List[Callable], None] = None
    test_transform: Union[List[Callable], None] = None
    predict_transform: Union[List[Callable], None] = None
    fit_dataloader_options: Dict = field(default_factory=lambda: DEFAULT_DATALOADER_OPTIONS)
    validate_dataloader_options: Dict = field(default_factory=lambda: DEFAULT_DATALOADER_OPTIONS)
    test_dataloader_options: Dict = field(default_factory=lambda: DEFAULT_DATALOADER_OPTIONS)
    predict_dataloader_options: Dict = field(default_factory=lambda: DEFAULT_DATALOADER_OPTIONS)
    train_val_split: float = 0.8
    by_zone: Union[None, str, List[STAGES]] = None
    patch_size: int = 256
    patch_resolution: List[float] = field(default_factory=lambda: [0.2, 0.2])
    random_window: bool = True
    overlap: Overlap = 0.0
    cache_dataset: Union[None, str, List[STAGES]] = False
    _data_loaders: Dict[STAGES, DataLoader] = field(init=False, default_factory=lambda: dict())
    _fit_df: DATAFRAME = field(init=False)
    _validate_df: DATAFRAME = field(init=False)
    _test_df: DATAFRAME = field(init=False)
    _predict_df: DATAFRAME = field(init=False)
    _fit_dataset: Dataset = field(init=False)
    _validate_dataset: Dataset = field(init=False)
    _test_dataset: Dataset = field(init=False)
    _predict_dataset: Dataset = field(init=False)
    _fit_transforms: Callable = field(init=False)
    _validate_transforms: Callable = field(init=False)
    _test_transforms: Callable = field(init=False)
    _predict_transforms: Callable = field(init=False)

    def _stage_by_zone(self, stage) -> bool:
        if self.by_zone == stage or self.by_zone == "all" or stage in self.by_zone:
            return True
        else:
            return False

    def _get_root_dir_for_stage(self, stage: STAGES) -> Optional[str]:
        if isinstance(self.root_dir, str) or self.root_dir is None:
            return self.root_dir
        else:
            if stage in self.root_dir.keys():
                return self.root_dir[stage]
            else:
                return None

    def _get_cache_preproc_status_by_stage(self, stage: STAGES) -> bool:
        if self. cache_dataset == stage or self.cache_dataset == "all" or stage in self.cache_dataset:
            return True
        else:
            return False

    def _input_file_has_header(self, stage: STAGES):
        if isinstance(self.input_files_has_header, bool):
            return self.input_files_has_header
        elif stage in self.input_files_has_header.keys():
            return self.input_files_has_header[stage]
        else:
            return True

    def __post_init__(self):

        if self.input_fit_file is not None:

            self._fit_df = create_dataframe_from_file(self.input_fit_file,
                                                      {"header": self._input_file_has_header(Stages.FIT)})
            self._fit_transforms = AlbuTransform(input_fields=self.input_fields,
                                                 pipe=self.fit_transform)
            self._fit_dataset = UniversalDataset(data=self._fit_df,
                                                 input_fields=self.input_fields,
                                                 transform=self._fit_transforms,
                                                 by_zone=self._stage_by_zone(Stages.FIT),
                                                 patch_size=self.patch_size,
                                                 patch_resolution=self.patch_resolution,
                                                 random_window=self.random_window,
                                                 inference_mode=False)
            self._data_loaders[Stages.FIT] = DataLoader(dataset=self._fit_dataset,
                                                        **self.fit_dataloader_options)

        if self.input_validate_file is None and self.input_fit_file:

            self._fit_df, self._validate_df = split_dataframe(self._fit_df, split_ratio=self.train_val_split)
            self._validate_transforms = AlbuTransform(input_fields=self.input_fields,
                                                      pipe=self.validate_transform)
            self._validate_dataset = UniversalDataset(data=self._validate_df,
                                                      input_fields=self.input_fields,
                                                      transform=self._validate_transforms)
            self._data_loaders[Stages.VALIDATE] = DataLoader(dataset=self._validate_dataset,
                                                             **self.validate_dataloader_options)

        if self.input_validate_file is not None:

            self._validate_df = create_dataframe_from_file(self.input_validate_file)
            self._validate_transforms = AlbuTransform(input_fields=self.input_fields,
                                                      pipe=self.validate_transform)
            self._validate_dataset = UniversalDataset(data=self._validate_df,
                                                      input_fields=self.input_fields,
                                                      transform=self._validate_transforms)
            self._data_loaders[Stages.VALIDATE] = DataLoader(dataset=self._validate_dataset,
                                                             **self.validate_dataloader_options)

        if self.input_test_file is not None:

            self._test_df = create_dataframe_from_file(self.input_test_file)
            self._test_transforms = AlbuTransform(input_fields=self.input_fields,
                                                  pipe=self.test_transform)
            self._test_dataset = UniversalDataset(data=self._test_df,
                                                  input_fields=self.input_fields,
                                                  transform=self._test_transforms)
            self._data_loaders[Stages.TEST] = DataLoader(dataset=self._test_dataset,
                                                         **self.test_dataloader_options)

        if self.input_predict_file is not None:

            self._predict_df = create_dataframe_from_file(self.input_predict_file)
            self._predict_transforms = AlbuTransform(input_fields=self.input_fields,
                                                     pipe=self.predict_transform)
            self._predict_dataset = UniversalDataset(data=self._predict_df,
                                                     input_fields=self.input_fields,
                                                     transform=self._predict_transforms)
            self._data_loaders[Stages.FIT] = DataLoader(dataset=self._predict_dataset,
                                                        **self.predict_dataloader_options)

    def get_dataloader(self, stage: STAGES):
        return self._data_loaders[stage]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.get_dataloader(Stages.FIT)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(Stages.VALIDATE)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(Stages.TEST)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(Stages.PREDICT)
