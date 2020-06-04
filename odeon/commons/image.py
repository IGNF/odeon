import gdal
import numpy as np
import logging

logger = logging.getLogger(__package__)

def image_to_ndarray(image_file, width=None, height=None, band_indices=None):
    """Load and transform an image into a ndarray according to parameters:
    - center-cropping to fit width, height
    - loading of specific bands to fit image_bands array

    Parameters
    ----------
    image_file : str
        full image file path
    width : int, optional
        output image width, by default None (native image width is used)
    height : int, optional
        output image height, by default None (native image height is used)
    band_indices : :obj:`list` of :obj: `int`, optional
        list of band indices to be loaded in output image, by default None (native image bands are used)

    Returns
    -------
    out: ndarray
        a numpy array representing the image
    """

    # load full image at a specific resolution
    ds = gdal.Open(image_file, gdal.GA_ReadOnly)
    if ds is None:
        logger.error(f"File {image_file} not valid.")

    # center crop with width and height
    img = ds.ReadAsArray()
    dims = len(img.shape)  # one channel or many

    if dims == 3:
        img = img.swapaxes(0, 2).swapaxes(0, 1)
    else:
        img = img[..., np.newaxis]
    # image shape is now W x H x B

    if width is None:
        width = img.shape[0]
    if height is None:
        height = width

    # cropping
    crop_size_x = img.shape[0] - width
    crop_size_y = img.shape[1] - height
    dx_left = int(crop_size_x / 2)
    dx_right = crop_size_x - dx_left
    dy_top = int(crop_size_y / 2)
    dy_bottom = crop_size_y - dy_top

    if not (dx_left == 0 or dx_right == 0):
        img = img[dx_left:-dx_right, :, :]
    if not (dy_top == 0 or dy_bottom == 0):
        img = img[:, dy_top:-dy_bottom, :]

    # bands selection
    if dims == 3 and not(band_indices is None):
        img = img[:, :, band_indices]

    return img
