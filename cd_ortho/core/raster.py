import numpy as np
import rasterio as rio


def read(src: rio.DatasetReader,
         band_indices=None,
         window=None,
         boundless: bool = None) -> np.ndarray:
    """

    Parameters
    ----------
    src
    band_indices
    window
    boundless

    Returns
    -------

    """
    if band_indices is None:
        band_indices = range(1, src.count + 1)

    if window is None:
        img = src.read(indexes=band_indices)
    else:
        img = src.read(indexes=band_indices,
                       window=window,
                       boundless=boundless)
    return img
