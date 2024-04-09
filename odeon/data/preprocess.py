"""Preprocess module, handles data features inside A Dataset class"""
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rasterio.plot import reshape_as_image
from rasterio.windows import from_bounds as window_from_bounds
from torch import Tensor

from odeon.core.types import URI
from odeon.layers.data import DTYPE_MAX, InputDType
from odeon.layers.rio import get_dataset, read

logger = getLogger(__name__)
MEAN_DEFAULT_VALUE = 0.5
STD_DEFAULT_VALUE = 0.5


class PreProcessor:
    """
    A preprocessing module that handles data features within a Dataset class, particularly focusing
    on raster and mask data. It supports operations like normalization, application of transformations,
    and caching of datasets for optimized data handling.

    Parameters
    ----------
    input_fields : Dict
        A dictionary specifying the input fields and their attributes, such as name, type (e.g., 'raster'
        or 'mask'), band indices, and normalization parameters.
    patch_size : int, optional
        The size of the patches to extract from raster data, defaults to 256.
    root_dir : Union[None, Path, str], optional
        The root directory from where to load the data files, defaults to None.
    cache_dataset : bool, optional
        If True, the dataset will be cached to improve performance, defaults to False.

    Methods
    -------
    forward(data: Dict, bounds: Optional[List] = None) -> Dict
        Processes the input data according to the configuration specified in `input_fields`. It
        supports operations such as reading and normalizing raster data, reading mask data, and
        optionally applying transformations.
    apply_to_raster(path: URI, band_indices: Optional[List] = None, bounds: Optional[List] = None,
                    dtype_max: float = DTYPE_MAX[InputDType.UINT8.value], mean: Optional[List] = None,
                    std: Optional[List] = None) -> np.ndarray
        Reads and processes raster data from a given path, applying normalization and returning
        the processed numpy array.
    apply_to_mask(path: URI, band_indices: Optional[List] = None, bounds: Optional[List] = None,
                  one_hot_encoding: bool = False) -> np.ndarray
        Reads and processes mask data from a given path, applying optional one-hot encoding and
        returning the processed numpy array.

    Examples
    --------
    >>> input_fields = {
    ...     "raster1": {"name": "image.tif", "type": "raster", "band_indices": [1, 2, 3], "dtype": "uint8"},
    ...     "mask1": {"name": "mask.tif", "type": "mask", "one_hot_encoding": True}
    ... }
    >>> preprocessor = PreProcessor(input_fields=input_fields, patch_size=512, root_dir="/path/to/data")
    >>> data = {"image.tif": "image.tif", "mask.tif": "mask.tif"}
    >>> processed_data = preprocessor(data)

    Notes
    -----
    - The class is designed to work seamlessly within a dataset class like `OdnDataset` to handle
      preprocessing of geospatial data for machine learning models.
    - It supports flexible configurations through `input_fields` to accommodate various data types
      and preprocessing needs.
    """
    def __init__(self,
                 input_fields: Dict,
                 patch_size: int = 256,
                 root_dir: Union[None, Path, str] = None,
                 cache_dataset: bool = False,
                 *args,
                 **kwargs):
        self._input_fields: Dict = input_fields
        self._sanitize_input_fields()  # make sure input is compatible
        self.root_dir = root_dir
        self.cache_dataset = cache_dataset
        self.patch_size = patch_size
        self._cache: Optional[Dict] = dict() if self.cache_dataset else None
        self._sanitize_input_fields()

    def _sanitize_input_fields(self):
        for value in self._input_fields.values():
            assert "name" in value
            assert "type" in value

    def forward(self, data: Dict, bounds: Optional[List] = None) -> Dict:
        output_dict = dict()
        # window = None  # used to cache the first computation of window to not repeat unnecessary computation
        if bounds:
            output_dict["bounds"] = np.array(bounds)
        for key, value in self._input_fields.items():
            if value["type"] == "raster":
                path = data[value["name"]] if self.root_dir is None else Path(str(self.root_dir)) / data[value["name"]]
                band_indices = value["band_indices"] if "band_indices" in value else None
                dtype_max = value["dtype_max"] if "dtype_max" in value else None
                mean = value["mean"] if "mean" in value else None
                std = value["std"] if "std" in value else None
                if dtype_max is None and "dtype" in value:
                    dtype = value["dtype"]
                    if dtype in DTYPE_MAX.keys():
                        dtype_max = DTYPE_MAX[dtype]
                    else:
                        raise KeyError(f'your dtype  {dtype} for key {key} in your input_fields is not compatible')
                dtype_max = DTYPE_MAX[InputDType.UINT8.value] if dtype_max is None else dtype_max
                output_dict[key] = self.apply_to_raster(path=path,
                                                        band_indices=band_indices,
                                                        bounds=bounds,
                                                        dtype_max=dtype_max,
                                                        mean=mean,
                                                        std=std)
            if value["type"] == "mask":
                path = data[value["name"]] if self.root_dir is None else Path(str(self.root_dir)) / data[value["name"]]
                band_indices = value["band_indices"] if "band_indices" in value else None
                one_hot_encoding = value["one_hot_encoding"] if "one_hot_encoding" in value else False
                output_dict[key] = self.apply_to_mask(path=path,
                                                      band_indices=band_indices,
                                                      bounds=bounds,
                                                      one_hot_encoding=one_hot_encoding)
            if "geometry" in data.keys():
                output_dict["geometry"] = np.array(data["geometry"].bounds)
        return output_dict

    def __call__(self, data: Dict, bounds: Optional[List] = None, *args, **kwargs) -> Dict:
        return self.forward(data=data, bounds=bounds)

    def apply_to_raster(self,
                        path: URI,
                        band_indices: Optional[List] = None,
                        bounds: Optional[List] = None,
                        dtype_max: float = DTYPE_MAX[InputDType.UINT8.value],
                        mean: Optional[List] = None,
                        std: Optional[List] = None) -> np.ndarray:

        # TODO Could be interesting to optimize window computation (DON'T REPEAT COMPUTATION FOR EACH MODALITY)
        # TODO But it could bring side effect, needs reflection
        src, window = self._get_dataset(path=path, bounds=bounds)
        raster = read(src, band_indices=band_indices, window=window, height=self.patch_size, width=self.patch_size)
        if self.cache_dataset is False:
            src.close()

        # add axe if raster has only two dimensions
        if raster.ndim == 2:
            raster = raster[..., np.newaxis]
        img = reshape_as_image(raster)
        if mean is None or std is None:
            # apply centering between 0 and 1
            # to apply centering between -1 and 1, replace change MEAN_DEFAULT_VALUE value to 0.5
            mean = [float(MEAN_DEFAULT_VALUE) for _ in range(raster.shape[0])]
            assert mean is not None
            std = [float(MEAN_DEFAULT_VALUE) for _ in range(raster.shape[0])]
            assert std is not None

        return normalize(img=img,
                         mean=mean,
                         std=std,
                         max_pixel_value=dtype_max)

    def apply_to_mask(self,
                      path: URI,
                      band_indices: Optional[List] = None,
                      bounds: Optional[List] = None,
                      one_hot_encoding: bool = False
                      ) -> np.ndarray:
        # TODO Could be interesting to optimize window computation (DON'T REPEAT COMPUTATION FOR EACH MODALITY)
        # TODO but it bring side effect
        src, window = self._get_dataset(path=path, bounds=bounds)
        mask = read(src, band_indices=band_indices, window=window, height=self.patch_size, width=self.patch_size)
        if self.cache_dataset is False:
            src.close()
        if one_hot_encoding:
            mask = np.argmax(mask, axis=0)
            mask = np.expand_dims(mask, axis=0)
        mask = reshape_as_image(mask)

        return mask

    def _get_dataset(self,
                     path: Union[str, Path],
                     bounds: Optional[List] = None
                     ) -> Tuple[Any, Any]:

        src = get_dataset(src=str(path), cached=self.cache_dataset)
        meta = src.meta
        window = None if bounds is None else window_from_bounds(bounds[0],
                                                                bounds[1],
                                                                bounds[2],
                                                                bounds[3],
                                                                meta["transform"])
        return src, window

    @property
    def cache(self):
        return self._cache


class UniversalDeProcessor:

    def __init__(self,
                 input_fields: Dict):
        self._input_fields = input_fields


def normalize(img: np.ndarray,
              mean: List[float],
              std: List[float],
              max_pixel_value: float = float(DTYPE_MAX[InputDType.UINT8.value])) -> np.ndarray:
    """
    Normalize an image with mean and std
    Parameters
    ----------
    img : np.ndarray
    mean : List[float]
    std : List[float]
    max_pixel_value : float

    Returns
    -------

    """
    mean_np = np.array(mean, dtype=np.float32)
    mean_np *= max_pixel_value
    std_np = np.array(std, dtype=np.float32)
    std_np *= max_pixel_value
    denominator = np.reciprocal(std_np, dtype=np.float32)
    img = img.astype(np.float32)
    img -= mean_np
    img *= denominator
    return img


def denormalize_tensor(image: Tensor, mean: List[float], std: List[float]):
    """
    Denormalize a tensor image with mean and standard deviation.
    Parameters
    ----------
    image : Tensor
    mean : List[float]
    std : List[float]

    Returns
    -------
    Tensor
    """
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)

    return image
