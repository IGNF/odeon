from typing import Optional, Sequence

import numpy as np
import rasterio as rio


def read(src: rio.DatasetReader,
         band_indices=None,
         window: Optional[Sequence] = None,
         boundless: bool = None,
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
