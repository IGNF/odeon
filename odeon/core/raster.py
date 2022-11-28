from functools import cache
from typing import Callable, Optional

import numpy as np
import rasterio as rio
from rasterio.windows import Window

from .types import URI


def read(src: rio.DatasetReader,
         band_indices=None,
         window: Optional[Window] = None,
         boundless: bool = True,
         height: Optional[int] = None,
         width: Optional[int] = None,
         resampling: rio.enums.Resampling = rio.enums.Resampling.nearest) -> np.ndarray:
    """

    Parameters
    ----------
    src
    band_indices
    window
    boundless
    height
    width
    resampling

    Returns
    -------

    """
    if band_indices is None:
        band_indices = range(1, src.count + 1)

    if window is None:
        img = src.read(indexes=band_indices)
    else:
        if height is not None and width is not None:
            img = src.read(indexes=band_indices,
                           window=window,
                           boundless=boundless,
                           out_shape=(len(band_indices), height, width),
                           resampling=resampling)
        else:
            img = src.read(indexes=band_indices,
                           window=window,
                           boundless=boundless)
    return img


def get_no_cache_dataset(src: URI) -> rio.DatasetReader:
    return rio.open(src)


@cache
def get_cached_dataset(src: URI) -> rio.DatasetReader:
    return rio.open(src)


def get_dataset(src: URI, cached: bool = False) -> rio.DatasetReader:
    if cached:
        return get_cached_dataset(src)
    else:
        return get_no_cache_dataset(src)


class RioDataset:
    def __init__(self, fn_read: Callable[[URI], rio.open]):
        self.fn_read: Callable = fn_read
        self.ds: rio.DatasetReader

    def __enter__(self, uri: URI):
        self.ds = self.fn_read(uri)
        return self.ds

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ds.close()
