from functools import cache
from typing import Callable, Optional

import numpy as np
import rasterio as rio
from rasterio.windows import Window

from odeon.core.types import URI

from ._engine import Engine
from .types import BOUNDS


def write_raster(data: np.ndarray,
                 path: str,
                 driver: str = 'GTiff',
                 bounds: Optional[BOUNDS] = None,
                 dtype: rio.dtypes | str = rio.uint8,
                 width: int = None, height: int = None,
                 count: int = None, crs=None,
                 transform=None,
                 compression: str = 'lzw',
                 tiled: bool = False,
                 blockxsize: int = 256, blockysize: int = 256,
                 interleave: str = 'band',
                 nodata: float = None,
                 photometric: str = 'RGB',
                 metadata: dict = None) -> None:
    """
    Writes numpy array data to a raster file with the specified options.

    Parameters
    ----------
    data : np.ndarray
        The array data to write.
    path : str
        Destination URI where the raster will be saved.
    driver : str
        Rasterio driver to use for writing the file.
    bounds : tuple, optional, default: None
        optional bounds of the raster to be written in.
    dtype : rasterio.dtypes or str
        Data type of the output file.
    width, height : int
        Dimensions of the raster.
    count : int
        Number of bands in the raster.
    crs :
        Coordinate reference system of the raster.
    transform :
        Affine transform for the raster (georeferencing).
    compression : str
        Compression type for the file.
    tiled : bool
        Whether to write the raster in tiled format.
    blockxsize, blockysize : int
        Block sizes for tiling.
    interleave : str
        Pixel interleave method (band, pixel).
    nodata : float
        No data value for the raster.
    photometric : str
        Photometric interpretation.
    metadata : dict
        Additional metadata for the raster file.
    """
    if width is None or height is None:
        raise ValueError("Width and height must be specified for the raster.")
    if count is None:
        count = data.shape[0] if data.ndim > 2 else 1  # Assuming data is in the shape [bands, rows, cols]
    with rio.open(path, 'w', driver=driver, height=height, width=width,
                  count=count, dtype=dtype, crs=crs, transform=transform,
                  compression=compression, tiled=tiled, blockxsize=blockxsize,
                  blockysize=blockysize, interleave=interleave, nodata=nodata,
                  photometric=photometric) as dst:
        if data.ndim == 3:  # Multiple bands
            for i in range(count):
                dst.write(data[i, :, :], i + 1)
        else:  # Single band
            dst.write(data, 1)

        if metadata is not None:
            dst.update_tags(**metadata)


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
        if height and width:
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


class RioEngine(Engine):

    def is_geo_referenced(self) -> bool:
        return True

    @staticmethod
    def read(src: rio.DatasetReader,
             band_indices=None,
             bounds: Optional[BOUNDS] = None,
             boundless: bool = True,
             height: Optional[int] = None,
             width: Optional[int] = None,
             resampling: rio.enums.Resampling = rio.enums.Resampling.nearest,
             ) -> np.ndarray:

        window = None
        return read(src=src, band_indices=band_indices, window=window, boundless=boundless,
                    height=height, width=width, resampling=resampling)

    def write(self, *args, **kwargs):
        ...
