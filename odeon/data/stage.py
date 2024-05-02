from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from layers.core.types import DATAFRAME
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from odeon.core.app_utils import Stages
from odeon.core.exceptions import MisconfigurationException
from odeon.core.types import STAGES_OR_VALUE, URI, GeoTuple
from odeon.data.core.dataloader_utils import (
    DEFAULT_DATALOADER_OPTIONS, DEFAULT_INFERENCE_DATALOADER_OPTIONS,
    DEFAULT_OVERLAP, DEFAULT_PATCH_RESOLUTION, DEFAULT_PATCH_SIZE)
from odeon.layers.dataframe import create_dataframe_from_file

from .dataset import OdnDataset
from .transform import AlbuTransform


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False, slots=True)
class DataFactory:
    """
        A factory class for generating DataLoader, Dataset, transformation functions, and DataFrame
        objects configured for different stages (fit, validate, test, predict) in a machine learning
        workflow. It is specially designed for handling geospatial data and supports a wide range of
        customization options for data loading and augmentation.

        Parameters
        ----------
        stage : STAGES_OR_VALUE
            The stage of the machine learning workflow for which data is being prepared. This affects
            the inference mode and possibly other configurations.
        input_file : URI
            The file path or URI to the input data.
        input_fields : Dict
            A dictionary mapping from field names to data types or other relevant specifications, used
            to structure the input data.
        by_zone : bool, optional
            Whether to process data by geographical zone, defaults to False.
        transforms : Union[List[Callable], None], optional
            A list of transformation functions or callables to be applied to the data, defaults to None.
        dataloader_options : Dict, optional
            Options and configurations for the DataLoader, defaults to an empty dictionary.
        root_dir : Optional[URI], optional
            The root directory for relative paths in the data, defaults to None.
        header : bool | str | None, optional
            Indicates if the input file has a header. Can be a boolean, a string specifying the header
            line, or 'infer' to automatically detect, defaults to 'infer'.
        header_list : List[str] | None, optional
            A list of header names to use for the DataFrame, defaults to None.
        patch_size : Union[int, Tuple[int, int], List[int]], optional
            The size of patches to be extracted from the input images, defaults to DEFAULT_PATCH_SIZE.
        patch_resolution : Union[float, Tuple[float, float], List[float]], optional
            The resolution of patches to be extracted, defaults to DEFAULT_PATCH_RESOLUTION.
        random_window : bool, optional
            Whether to use random windows for patch extraction, defaults to True.
        overlap : Union[GeoTuple], optional
            The overlap between patches, if any, defaults to DEFAULT_OVERLAP.
        cache_dataset : Union[bool], optional
            Whether to cache the dataset, defaults to False.
        debug : bool, optional
            If True, enables debug mode with more verbose logging, defaults to False.

        Methods
        -------
        build_data(...)
            A class method that instantiates a `DataFactory` object and returns DataLoader, Dataset,
            transformation functions, and DataFrame objects configured according to the provided
            parameters.

        Examples
        --------
        >>> stage = Stages.FIT
        >>> input_file = 'path/to/data.csv'
        >>> input_fields = {'field1': 'float', 'field2': 'int'}
        >>> transforms = [lambda x: x + 1, lambda x: x + 2]
        >>> data_loader, dataset, transform, dataframe = DataFactory.build_data(
        ...     stage=stage,
        ...     input_file=input_file,
        ...     input_fields=input_fields,
        ...     transform=transforms
        ... )

        Notes
        -----
        This class provides a powerful interface for configuring and generating the data infrastructure
        needed for training, validating, testing, and predicting in machine learning models, especially
        those dealing with geospatial data. It allows for extensive customization to suit various data
        formats, preprocessing needs, and workflow requirements.
        """
    stage: STAGES_OR_VALUE
    input_file: URI
    input_fields: Dict
    by_zone: bool = False
    transforms: Union[List[Callable], None] = None
    dataloader_options: Dict = field(default_factory=lambda: {})
    root_dir: Optional[URI] = None
    header: bool | str | None = 'infer'  # Rather input files have header or not
    header_list: List[str] | None = None
    patch_size: Union[int, Tuple[int, int], List[int]] = field(default_factory=lambda: DEFAULT_PATCH_SIZE)
    patch_resolution: Union[float, Tuple[float, float], List[float]] = field(
        default_factory=lambda: DEFAULT_PATCH_RESOLUTION)
    random_window: bool = True
    overlap: Union[GeoTuple] = field(default_factory=lambda: DEFAULT_OVERLAP)
    cache_dataset: Union[bool] = False
    debug: bool = False
    # TODO see how we could handle different crs, torchgeo style or another solution
    #  crs: Union[str, Dict, None] = "EPSG:2154"
    _dataloader: DataLoader = field(init=False)
    _dataframe: DATAFRAME = field(init=False)
    _dataset: Dataset = field(init=False)
    _transform: Optional[Callable] = field(init=False, default=None)
    _inference_mode: bool = field(init=False)
    _patch_size: Tuple[int, int] = field(init=False)

    def __post_init__(self):

        if isinstance(self.patch_size, (Tuple, List)):
            self._patch_size = (self.patch_size[0], self.patch_size[1])
        else:
            self._patch_size = (self.patch_size, self.patch_size)
        self._inference_mode = False if self.stage in [Stages.FIT, Stages.FIT.value] else True
        self._dataframe = create_dataframe_from_file(path=self.input_file,
                                                     options={'header': self.header,
                                                              'header_list': self.header_list})
        self._transform = AlbuTransform(input_fields=self.input_fields,
                                        pipe=self.transforms)
        assert self._transform is not None
        self._dataset = OdnDataset(input_fields=self.input_fields,
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
                   header: bool | str | None = "infer",
                   header_list: List[str] | None = None,
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
                           header=header,
                           header_list=header_list,
                           by_zone=by_zone)
        return data_factory.dataloader, data_factory.dataset, data_factory.transform, data_factory.dataframe
