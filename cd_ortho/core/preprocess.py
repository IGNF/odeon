"""Preprocess module, handles data preprocessing inside A Dataset class"""
from typing import Dict, List, Optional

import numpy as np

from .data import DTYPE_MAX, InputDType
from .raster import read, rio

MEAN_DEFAULT_VALUE = 0.0
STD_DEFAULT_VALUE = 0.5


class UniversalPreProcessor:

    def __init__(self, input_fields: Dict, *args, **kwargs):
        self._input_fields: Dict = input_fields
        self._sanitize_input_fields()  # make sure input is compatible

    def _sanitize_input_fields(self):

        for value in self._input_fields.values():
            assert "name" in value
            assert "type" in value

    def forward(self, data: Dict) -> Dict:
        output_dict = dict()
        for key, value in self._input_fields.items():
            if value["type"] == "raster":

                path = data[value["name"]]
                band_indices = value["band_indices"] if "band_indices" in value else None
                dtype_max = value["dtype_max"] if "dtype_max" in value else None

                if dtype_max is None and "dtype" in value:
                    dtype = value["dtype"]
                    if dtype in DTYPE_MAX.keys():
                        dtype_max = DTYPE_MAX[dtype]
                    else:
                        raise KeyError(f'your dtype  {dtype} for key {key} in your input_fields is not compatible')
                dtype_max = DTYPE_MAX[InputDType.UINT8.value] if dtype_max is None else dtype_max
                output_dict[key] = UniversalPreProcessor.apply_to_raster(path=path,
                                                                         band_indices=band_indices,
                                                                         dtype_max=dtype_max)

            if value["type"] == "mask":
                path = data[value["name"]]
                output_dict[key] = UniversalPreProcessor.apply_to_mask(path=path)

        return output_dict

    def __call__(self, data: Dict, *args, **kwargs) -> Dict:
        return self.forward(data=data)

    @staticmethod
    def apply_to_raster(path, band_indices=None,
                        window=None,
                        dtype_max: float = DTYPE_MAX[InputDType.UINT8.value],
                        mean: Optional[List] = None,
                        std: Optional[List] = None):

        with rio.open(path) as src:
            raster = read(src, band_indices, window)
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

    @staticmethod
    def apply_to_mask(path):

        with rio.open(path) as src:
            mask = read(src)
        return mask


class UniversalDeProcessor:

    def __init__(self, input_fields):
        self._input_fields = input_fields

    def forward(self, data: Dict, *args, **kwargs) -> Dict:
        ...

    def __call__(self, data: Dict, *args, **kwargs) -> Dict:
        ...


def normalize(img, mean, std, max_pixel_value=float(DTYPE_MAX[InputDType.UINT8.value])):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img
