"""Preprocess module, handles data preprocessing inside A Dataset class"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .data import DTYPE_MAX, InputDType
from .raster import read, rio, window_from_bounds

MEAN_DEFAULT_VALUE = 0.0
STD_DEFAULT_VALUE = 0.5


class UniversalPreProcessor:

    def __init__(self,
                 input_fields: Dict,
                 patch_size: int = 256,
                 root_dir: Optional[str] = None,
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

        return output_dict

    def __call__(self, data: Dict, bounds: Optional[List] = None, *args, **kwargs) -> Dict:
        return self.forward(data=data, bounds=bounds)

    def apply_to_raster(self,
                        path: Union[str, Path],
                        band_indices: Optional[List] = None,
                        bounds: Optional[List] = None,
                        dtype_max: float = DTYPE_MAX[InputDType.UINT8.value],
                        mean: Optional[List] = None,
                        std: Optional[List] = None) -> np.ndarray:
        # TODO Could be interesting to optimize window computation (DON'T REPEAT COMPUTATION FOR EACH MODALITY)
        # TODO but it bring side effect
        src, window = self._get_dataset(path=path, bounds=bounds)
        raster = read(src, band_indices=band_indices, window=window, height=self.patch_size, width=self.patch_size)
        # add axe if raster has only two dimensions
        if raster.ndim == 2:
            raster = raster[..., np.newaxis]
        if mean is None or std is None:

            # apply centering between 0 and 1
            # to apply centering between -1 and 1, replace change MEAN_DEFAULT_VALUE value to 0.5
            mean = [MEAN_DEFAULT_VALUE]
            mean = list(np.repeat(mean, range(raster.shape[0])))
            std = [STD_DEFAULT_VALUE]
            std = list(np.repeat(std, range(raster.shape[0])))

        return normalize(raster,
                         mean=mean,
                         std=std,
                         max_pixel_value=dtype_max)

    def apply_to_mask(self,
                      path: Union[str, Path],
                      band_indices: Optional[List] = None,
                      bounds: Optional[List] = None
                      ) -> np.ndarray:
        # TODO Could be interesting to optimize window computation (DON'T REPEAT COMPUTATION FOR EACH MODALITY)
        # TODO but it bring side effect
        src, window = self._get_dataset(path=path, bounds=bounds)
        mask = read(src, band_indices=band_indices, window=window, height=self.patch_size, width=self.patch_size)
        return mask

    def _get_dataset(self,
                     path: Union[str, Path],
                     bounds: Optional[List] = None
                     ) -> Tuple[Any, Any]:

        if self._cache is not None and path in self._cache.keys():
            src = self._cache[path]
            meta = src.meta
        else:
            with rio.open(path) as src:
                meta = src.meta
                if self._cache is not None:
                    self._cache[path] = src

        window = None if bounds is None else window_from_bounds(bounds[0],
                                                                bounds[1],
                                                                bounds[2],
                                                                bounds[3],
                                                                meta["transform"])
        return src, window


class UniversalDeProcessor:

    def __init__(self, input_fields):
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
