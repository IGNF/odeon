"""Preprocess module, handles data features inside A Dataset class"""
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rasterio.plot import reshape_as_image
from rasterio.windows import from_bounds as window_from_bounds
from torch import Tensor

from odeon.core.data import DTYPE_MAX, InputDType
from odeon.core.raster import get_dataset, read
from odeon.core.types import URI

logger = getLogger(__name__)
MEAN_DEFAULT_VALUE = 0.0
STD_DEFAULT_VALUE = 0.5


class UniversalPreProcessor:

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
                mean = value["mean"] if "band_indices" in value else None
                std = value["std"] if "band_indices" in value else None
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
                output_dict[key] = self.apply_to_mask(path=path,
                                                      band_indices=band_indices,
                                                      bounds=bounds)

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
            mean = [MEAN_DEFAULT_VALUE]
            mean = np.repeat(mean, raster.shape[0])
            std = [STD_DEFAULT_VALUE]
            std = np.repeat(std, raster.shape[0])

        return normalize(img=img,
                         mean=mean,
                         std=std,
                         max_pixel_value=dtype_max)

    def apply_to_mask(self,
                      path: URI,
                      band_indices: Optional[List] = None,
                      bounds: Optional[List] = None
                      ) -> np.ndarray:
        # TODO Could be interesting to optimize window computation (DON'T REPEAT COMPUTATION FOR EACH MODALITY)
        # TODO but it bring side effect
        src, window = self._get_dataset(path=path, bounds=bounds)
        mask = read(src, band_indices=band_indices, window=window, height=self.patch_size, width=self.patch_size)
        if self.cache_dataset is False:
            src.close()
        return reshape_as_image(mask)

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

    def forward(self, data: Dict, *args, **kwargs) -> Dict:
        ...

    def __call__(self, data: Dict, *args, **kwargs) -> Dict:
        ...


def normalize(img, mean, std, max_pixel_value=float(DTYPE_MAX[InputDType.UINT8.value])) -> np.ndarray:

    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value
    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value
    denominator = np.reciprocal(std, dtype=np.float32)
    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def denoromalize_img_as_tensor(image: Tensor, mean, std):
    """
            Args:
                tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            Returns:
                Tensor: Normalized image.
    """
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)

    return image
