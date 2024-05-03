from functools import cache
from typing import Callable, Dict, List, Optional

import numpy as np
import rasterio as rio
from rasterio.io import DatasetReaderBase, DatasetWriterBase
from rasterio.plot import reshape_as_raster
from rasterio.windows import Window

from odeon.core.types import PARAMS, URI

from .core.dtype import DType
from .core.engine import Engine
from .core.types import BOUNDS
from .rio_utils import get_read_window, get_write_transform, get_write_window


def _write_data(dst: DatasetWriterBase, data: np.ndarray, bounds: Optional[BOUNDS], masked: bool = True,
                width: int = None, height: int = None, as_raster: bool = False, count: int = 1
                ) -> None:
    if bounds and width and height:
        window = get_write_window(src=dst, bounds=bounds)
    else:
        window = None
    if data.ndim == 3:  # Multiple bands
        data = reshape_as_raster(data) if as_raster else data
        dst.write(data, indexes=[i for i in range(1, count + 1)], masked=masked, window=window)
    else:  # Single band
        dst.write(data, 1, window=window)


def write_raster(data: np.ndarray,
                 path: URI = None,
                 dst: Optional[DatasetWriterBase] = None,
                 bounds: Optional[BOUNDS] = None,
                 driver: str = 'GTiff',
                 mode: str = 'w',
                 dtype: DType | str = 'uint8',
                 width: int = None,
                 height: int = None,
                 count: int = None,
                 crs=None,
                 transform=None,
                 compression: str = 'lzw',
                 tiled: bool = False,
                 blockxsize: int = 256,
                 blockysize: int = 256,
                 interleave: str = 'band',
                 nodata: float = None,
                 photometric: str = 'RGB',
                 metadata: dict = None,
                 masked: bool = False) -> DatasetWriterBase:
    """
    Writes numpy array data to a raster file with the specified options.

    Parameters
    ----------
    data : np.ndarray
        The array data to write, in numpy image shape format (WxHxC)
    path : str or Pathlike, optional
        Destination URI where the raster will be saved.
    dst: DatasetWriterBase, optional
        rather than passing a URI, pass a Dataset writer object instead.
    driver : str
        Rasterio driver to use for writing the file.
    mode: str, default: 'w'
        'w' or 'w+', write mode
    bounds : tuple, optional, default: None
        optional bounds of the raster to be written in.
    dtype : rasterio.dtypes or str
        Data type of the output file.
    width, height : int
        Dimensions of the raster.
    count : int
        Number of bands in the raster.
    crs : rasterio.crs.CRS
        Coordinate reference system of the raster.
    transform : tuple
        Affine transform for the raster (georeferencing).
    compression : str
        Compression type for the file.
    tiled : bool
        Whether to write the raster in tiled format.
    blockxsize: int
        Block sizes x for tiling.
    blockysize : int
        Block sizes y for tiling.
    interleave : str
        Pixel interleave method (band, pixel).
    nodata : float
        No data value for the raster.
    photometric : str
        Photometric interpretation.
    metadata : dict
        Additional metadata for the raster file.
    masked : bool, optional
        Rather or not to write to the dataset's band mask.

    Returns
    -------
        DatasetWriterBase
    """
    assert dst is not None or path is not None, ('Either a destination URI path or a'
                                                 ' dst dataset writer object must be passed')
    if width is None or height is None:
        raise ValueError("Width and height must be specified for the raster.")
    if count is None:
        count = data.shape[0] if data.ndim > 2 else 1  # Assuming data is in the shape [bands, rows, cols]
    if dst is None:
        with rio.open(path, mode=mode, driver=driver, height=height, width=width,
                      count=count, dtype=dtype, crs=crs, transform=transform,
                      compression=compression, tiled=tiled, blockxsize=blockxsize,
                      blockysize=blockysize, interleave=interleave, nodata=nodata,
                      photometric=photometric) as dst:
            if metadata is not None:
                dst.update_tags(**metadata)
            _write_data(dst=dst, data=data, bounds=bounds, masked=masked, width=width, height=height)
            return dst
    else:
        _write_data(dst=dst, data=data, bounds=bounds, masked=masked, width=width, height=height)
        return dst


def read_raster(src: DatasetReaderBase,
                band_indices: Optional[int | List[int]] = None,
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
    np.ndarray
    """

    if band_indices is None:
        band_indices = [i for i in range(1, src.count + 1)]
    if window is None:
        img = src.read(indexes=band_indices)
    else:
        length = len(band_indices) if isinstance(band_indices, list) else 1
        if height and width:
            img = src.read(indexes=band_indices,
                           window=window,
                           boundless=boundless,
                           out_shape=(length, height, width),
                           resampling=resampling)
        else:
            img = src.read(indexes=band_indices,
                           window=window,
                           boundless=boundless)
    return img


def _get_no_cached_dataset(src: URI) -> DatasetReaderBase:
    return rio.open(src)


@cache
def _get_cached_dataset(src: URI) -> DatasetReaderBase:
    return rio.open(src)


def get_dataset(src: URI,
                cached: bool = False
                ) -> DatasetReaderBase:
    if cached:
        return _get_cached_dataset(src)
    else:
        return _get_no_cached_dataset(src)


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

    def __init__(self,
                 driver: str = 'GTiff',
                 mode: str = 'w',
                 dtype: DType | str = 'uint8',
                 width: int = None,
                 height: int = None,
                 count: int = None,
                 crs=None,
                 compression: str = 'lzw',
                 tiled: bool = False,
                 blockxsize: int = 256,
                 blockysize: int = 256,
                 interleave: str = 'band',
                 nodata: float = None,
                 photometric: str = 'RGB',
                 metadata: PARAMS = None,
                 masked: bool = False,
                 cache_dataset: bool = True,
                 boundless: bool = True,
                 resampling: rio.enums.Resampling = rio.enums.Resampling.nearest,
                 ):
        self.mode = mode
        self.driver = driver
        self.dtype = dtype
        self.width = width
        self.height = height
        self.count = count
        self.crs = crs
        self.compression = compression
        self.tiled = tiled
        self.blockxsize = blockxsize
        self.blockysize = blockysize
        self.interleave = interleave
        self.nodata = nodata
        self.photometric = photometric
        self.metadata = metadata
        self.masked = masked
        self.cache_dataset = cache_dataset
        self._cache: Dict[str, DatasetReaderBase] = {}
        self.boundless = boundless
        self.resampling = resampling

    def is_geo_referenced(self) -> bool:
        return True

    def read(self,
             path: Optional[URI] = None,
             src: Optional[DatasetReaderBase] = None,
             band_indices: Optional[List[int]] = None,
             bounds: Optional[BOUNDS] = None,
             ) -> np.ndarray:
        """
        Parameters
        ----------
        path: str or PathLike, optional, path of readable raster
        src: DatasetReaderBase, optional, data where to read data
        band_indices:
        bounds

        Returns
        -------
            np.ndarray
        """
        src = src if src is not None else get_dataset(str(path), cached=self.cache_dataset)
        window = get_read_window(bounds=bounds, transform=src.transform) if bounds is not None else None
        return read_raster(src=src, band_indices=band_indices, window=window, boundless=self.boundless,
                           height=self.height, width=self.width, resampling=self.resampling)

    def write(self,
              data: np.ndarray,
              bbox: Optional[BOUNDS] = None,
              path: Optional[URI] = None,
              bounds: Optional[BOUNDS] = None,
              width: Optional[int] = None,
              height: Optional[int] = None,
              *args, **kwargs):
        """ Write raster data to disk, in respect with the configuration of the Rio Engine

        Parameters
        ----------
        data: np.ndarray, data to write in rio raster
        path
        bbox: Bounding box, bounds of the output raster if no destination dataset has been set or cached
        path: URI, optional, path to write the output raster
        bounds: BOUNDS, optional, bounds of the data into the raster if we append data (engine must be in w+ mode)
        width: int, optional, width of the output raster
        height: int, optional, height of the output raster
        args
        kwargs

        Returns
        -------

        """
        width = width if width is not None else self.width
        height = height if height is not None else self.height
        # compute transform from bbox
        geo_id = str(path)
        if geo_id in self._cache:
            dst = self._cache[geo_id]
            transform = None
        else:
            transform = get_write_transform(bounds=bbox, width=width, height=height)
            dst = None
        # write data into dataset
        dst = write_raster(data=data, path=path, dst=dst, bounds=bounds, driver=self.driver, mode=self.mode,
                           dtype=self.dtype, width=width, height=height, count=self.count, crs=self.crs,
                           transform=transform, compression=self.compression, tiled=self.tiled,
                           blockxsize=self.blockxsize, blockysize=self.blockysize, interleave=self.interleave,
                           nodata=self.nodata, photometric=self.photometric, metadata=self.metadata,
                           masked=self.masked,)
        if self.cache_dataset:
            self._cache[geo_id] = dst

    def delete_cache(self):
        """close all cached datasets"""
        if len(self._cache) > 0:
            [dataset.close() for dataset in self._cache.values()]

    def __del__(self):
        self.delete_cache()
