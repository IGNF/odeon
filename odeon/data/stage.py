from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from odeon.core.dataframe import create_dataframe_from_file
from odeon.core.exceptions import MisconfigurationException
from odeon.core.app_utils import Stages
from odeon.core.types import DATAFRAME, STAGES_OR_VALUE, URI, GeoTuple
from odeon.data.core.dataloader_utils import (
    DEFAULT_DATALOADER_OPTIONS, DEFAULT_INFERENCE_DATALOADER_OPTIONS,
    DEFAULT_OVERLAP, DEFAULT_PATCH_RESOLUTION, DEFAULT_PATCH_SIZE)

from .dataset import UniversalDataset
from .transform import AlbuTransform


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False, slots=True)
class DataFactory:

    stage: STAGES_OR_VALUE
    input_file: URI
    input_fields: Dict
    transforms: Union[List[Callable], None] = None
    dataloader_options: Dict = field(default_factory=lambda: {})
    root_dir: Optional[URI] = None
    input_files_has_header: bool = True  # Rather input files have header or not
    by_zone: bool = False
    patch_size: Union[int, Tuple[int, int], List[int]] = field(default_factory=lambda: DEFAULT_PATCH_SIZE)
    patch_resolution: Union[float, Tuple[float,
                                         float], List[float]] = field(default_factory=lambda: DEFAULT_PATCH_RESOLUTION)
    random_window: bool = True
    overlap: Union[GeoTuple] = field(default_factory=lambda: DEFAULT_OVERLAP)
    cache_dataset: Union[bool] = False
    debug: bool = False
    # TODO see how we could handle different crs, torchgeo style or another solution
    #  crs: Union[str, Dict, None] = "EPSG:2154"
    _dataloader: DataLoader = field(init=False)
    _dataframe: DATAFRAME = field(init=False)
    _dataset: Dataset = field(init=False)
    _transform: Callable = field(init=False, default=None)
    _inference_mode: bool = field(init=False)
    _patch_size: Tuple[int, int] = field(init=False)

    def __post_init__(self):

        if isinstance(self.patch_size, (Tuple, List)):
            self._patch_size = (self.patch_size[0], self.patch_size[1])
        else:
            self._patch_size = (self.patch_size, self.patch_size)

        self._inference_mode = False if self.stage in [Stages.FIT, Stages.FIT.value] else True
        self._dataframe = create_dataframe_from_file(path=self.input_file,
                                                     options={'header': self.input_files_has_header})
        self._transform = AlbuTransform(input_fields=self.input_fields,
                                        pipe=self.transforms)
        self._dataset = UniversalDataset(input_fields=self.input_fields,
                                         data=self._dataframe,
                                         transform=self._transform,
                                         patch_resolution=self.patch_resolution,
                                         patch_size=self._patch_size,
                                         root_dir=self.root_dir,
                                         by_zone=self.by_zone,
                                         random_window=self.random_window,
                                         inference_mode=self._inference_mode,
                                         debug=self.debug,
                                         cache_dataset=self.cache_dataset
                                         )
        if self.dataloader_options == {}:
            self.dataloader_options = DEFAULT_INFERENCE_DATALOADER_OPTIONS if self._inference_mode \
                else DEFAULT_DATALOADER_OPTIONS

        if "weights" in self.dataloader_options:
            assert self.dataloader_options["weights"] in self._dataframe.columns
            weights = [float(i) for i in self._dataframe[self.dataloader_options["weights"]].tolist()]
            self.dataloader_options["sampler"] = WeightedRandomSampler(weights=weights, num_samples=len(weights))
            del self.dataloader_options["weights"]
        self._dataloader = DataLoader(dataset=self._dataset, **self.dataloader_options)

    @property
    def dataloader(self) -> DataLoader:
        return self._dataloader

    @dataloader.setter
    def dataloader(self, dataloader: DataLoader):
        self._dataloader = dataloader

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Dataset):
        self._dataset = dataset

    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe: DATAFRAME):
        self._dataframe = dataframe

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    @classmethod
    def build_data(cls,
                   stage: STAGES_OR_VALUE,
                   input_file: URI,
                   input_fields: Dict,
                   transform: Union[List[Callable], None] = None,
                   dataloader_options: Dict = None,
                   root_dir: Optional[URI] = None,
                   input_files_has_header: bool = True,
                   by_zone: bool = False,
                   patch_size: Union[int, Tuple[int, int], List[int]] = None,
                   patch_resolution: Union[float, Tuple[float, float], List[float]] = None,
                   random_window: bool = True,
                   overlap: Union[GeoTuple] = None,
                   cache_dataset: Union[bool] = False,
                   debug: bool = False) -> Tuple[DataLoader, Dataset, Callable, DATAFRAME]:
        if isinstance(stage, str) and stage not in [str(Stages.FIT.value),
                                                    str(Stages.VALIDATE.value),
                                                    str(Stages.TEST.value),
                                                    str(Stages.PREDICT.value)]:
            raise MisconfigurationException(message=f'stage {stage} is not recognized, should be one of '
                                                    f'{str(STAGES_OR_VALUE)}')
        dataloader_options = dataloader_options if dataloader_options is not None else {}
        patch_resolution = DEFAULT_PATCH_RESOLUTION if patch_resolution is None else patch_resolution
        patch_size = DEFAULT_PATCH_SIZE if patch_size is None else patch_size
        overlap = DEFAULT_OVERLAP if overlap is None else overlap
        data_factory = cls(stage=stage,
                           input_fields=input_fields,
                           input_file=input_file,
                           transforms=transform,
                           dataloader_options=dataloader_options,
                           patch_size=patch_size,
                           patch_resolution=patch_resolution,
                           overlap=overlap,
                           debug=debug,
                           cache_dataset=cache_dataset,
                           random_window=random_window,
                           root_dir=root_dir,
                           input_files_has_header=input_files_has_header,
                           by_zone=by_zone)
        return data_factory.dataloader, data_factory.dataset, data_factory.transform, data_factory.dataframe
