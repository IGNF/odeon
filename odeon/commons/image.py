import numpy as np
import rasterio
import math
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, transform
from rasterio.plot import reshape_as_image, reshape_as_raster
from skimage.transform import resize
from skimage import img_as_float
from odeon.commons.rasterio import get_scale_factor_and_img_size_from_dataset, get_center_from_bound
from odeon.commons.rasterio import get_bounds, create_patch_from_center
from odeon import LOGGER


def raster_to_ndarray_from_dataset(src,
                                   width,
                                   height,
                                   resolution,
                                   band_indices=None,
                                   resampling=Resampling.bilinear,
                                   window=None,
                                   boundless=True):

    """Load and transform an image into a ndarray according to parameters:
    - center-cropping to fit width, height
    - loading of specific bands to fit image_bands array

    Parameters
    ----------
    src : rasterio.DatasetReader
        raster source for the conversion
    width : int, optional
        output image width, by default None (native image width is used)
    height : int, optional
        output image height, by default None (native image height is used)
    resolution: obj:`list` of :obj: `number`
        output resolution for x and y
    band_indices : obj:`list` of :obj: `int`, optional
        list of band indices to be loaded in output image, by default None (native image bands are used)
    resampling: one enum from rasterio.Reampling
        resampling method to use when a resolution change is necessary.
        Default: Resampling.bilinear
    window: rasterio.window, see rasterio docs
        use a window in rasterio format or not to select a subsection of the raster
        Default: None
    Returns
    -------
    out: Tuple[ndarray, dict]
        a numpy array representing the image, and the metadata in rasterio format.
    """

    " get the width and height at the target resolution "
    _, _, scaled_width, scaled_height = get_scale_factor_and_img_size_from_dataset(src,
                                                                                   resolution=resolution,
                                                                                   width=width,
                                                                                   height=height)

    scaled_height, scaled_width = math.ceil(scaled_height), math.ceil(scaled_width)

    if band_indices is None:

        band_indices = range(1, src.count + 1)

    if window is None:

        img = src.read(indexes=band_indices,
                       out_shape=(len(band_indices), scaled_height, scaled_width),
                       resampling=resampling)
    else:

        img = src.read(indexes=band_indices,
                       window=window,
                       out_shape=(len(band_indices), scaled_height, scaled_width),
                       resampling=resampling,
                       boundless=boundless)

    " reshape img from gdal band format to numpy ndarray format "
    img = reshape_as_image(img)

    if img.ndim == 2:
        img = img[..., np.newaxis]

    if scaled_width >= width and scaled_height >= height:

        if scaled_width > width or scaled_height > height:
            " handling case of dest_res > src_res, example from 0.5 to 0.2 "
            img = crop_center(img, width, height)

    if scaled_width < width or scaled_height > height:
        " handling case of dest_res < src_res, example from 0.2 to 0.5 "
        img = resize(img, (width, height))

    meta = src.meta.copy()

    " compute the new affine transform "
    if scaled_width != width or scaled_height != height:

        side_x = (img.shape[0] * resolution[0]) / 2
        side_y = (img.shape[1] * resolution[1]) / 2
        bounds = src.bounds
        center_x, center_y = get_center_from_bound(bounds.left, bounds.bottom, bounds.right, bounds.top)
        left, bottom, right, top = center_x - side_x, center_y - side_y, center_x + side_x, center_y + side_y
        window = from_bounds(left, bottom, right, top, src.transform)
        affine = transform(window, src.transform)
        meta["transform"] = affine

    return img, meta


def raster_to_ndarray(image_file,
                      width,
                      height,
                      resolution,
                      band_indices=None,
                      resampling=Resampling.bilinear,
                      window=None):
    """Simple helper function to call raster_to_ndarray_from_dataset from
    a raster path file contrary to a rasterio.Dataset

    Parameters
    ----------
    src : rasterio.DatasetReader
        raster source for the conversion
    width : int, optional
        output image width, by default None (native image width is used)
    height : int, optional
        output image height, by default None (native image height is used)
    resolution: obj:`list` of :obj: `float`
        output resolution for x and y
    band_indices : obj:`list` of :obj: `int`, optional
        list of band indices to be loaded in output image, by default None (native image bands are used)
    resampling: one enum from rasterio.Reampling
        resampling method to use when a resolution change is necessary.
        Default: Resampling.bilinear
    window: rasterio.window, see rasterio docs
        use a window in rasterio format or not to select a subsection of the raster
        Default: None
    Returns
    -------
    out: Tuple[ndarray, dict]
        a numpy array representing the image, and the metadata in rasterio format.

    """
    with rasterio.open(image_file) as src:

        if resolution is None:
            resolution = src.res
        if width is None:
            width = src.width
        if height is None:
            height = src.height

        return raster_to_ndarray_from_dataset(src, width, height, resolution, band_indices, resampling, window)


def crop_center(img, cropx, cropy):
    """Crop numpy array based on the center of
    array (rounded to inf element for array of even size)
    We expect array to be of format W*H*C
    Parameters
    ----------
    img : numpy NDArray
        numpy array of dimension 2 or 3
    cropx : int
        size of crop on x axis (first axis)
    cropy : int
        size of crop on x axis (second axis)

    Returns
    -------
    numpy NDArray
        the cropped numpy 3d array
    """

    if img.ndim == 2:

        y, x = img.shape

    else:

        y, x, _ = img.shape

    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def substract_margin(img, margin_x, margin_y):
    """Crop numpy array based on the center of
    array (rounded to inf element for array of even size)
    We expect array to be of format H*W*C
    Parameters
    ----------
    img : numpy NDArray
        numpy array of dimension 2 or 3
    margin_x : int
        size of crop on x axis (first axis)
    margin_y : int
        size of crop on x axis (second axis)

    Returns
    -------
    numpy NDArray
        the substracted of its margin numpy 3darray
    """
    if img.ndim == 2:

        y, x = img.shape

    else:

        y, x, _ = img.shape

    return img[0 + margin_y: y - margin_y, 0 + margin_x:x - margin_x]


class TypeConverter:
    """Simple class to handle conversion of output format
    in detection notably.
    """
    def __init__(self):

        self._from = "float32"
        self._to = "uint8"

    def from_type(self, img_type):
        """get orignal type

        Parameters
        ----------
        img_type : str
            we actually handle only the case of 32 bits

        Returns
        -------
        TypeConverter
            itself
        """

        self._from = img_type
        return self

    def to_type(self, img_type):
        """get targer type

        Parameters
        ----------
        img_type : str
            we actually handle float32, int8, and thresholded
            1bit.
        Returns
        -------
        TypeConverter
            itself
        """

        self._to = img_type
        return self

    def convert(self, img, threshold=0.5):
        """Make conversion

        Parameters
        ----------
        img : NDArray
            input image
        threshold : float, optional
            used with 1bit output, to binarize pixels, by default 0.5

        Returns
        -------
        NDArray
            converted image
        """

        if self._from == "float32":

            if self._to == "float32":

                return img

            elif self._to == "uint8":

                if img.max() > 1:

                    info = np.idebug(img.dtype)  # Get the information of the incoming image type
                    img = img.astype(np.float32) / info.max  # normalize the data to 0 - 1

                img = 255 * img  # scale by 255
                return img.astype(np.uint8)

            elif self._to == "bit":

                img = img > threshold
                return img.astype(np.uint8)

            else:

                LOGGER.warning("the output type has not been interpreted")
                return img


def ndarray_to_affine(affine):
    """

    Parameters
    ----------
    affine : numpy Array

    Returns
    -------
    rasterio.Affine
    """
    return rasterio.Affine(affine[0], affine[1], affine[2], affine[3], affine[4], affine[5])


class CollectionDatasetReader:
    """Static class to handle connection fo multiple raster input

    Returns
    -------
    NDArray
        stacked bands of multiple rasters
    """
    @staticmethod
    def get_stacked_window_collection(dict_of_raster,
                                      bounds,
                                      width,
                                      height,
                                      resolution,
                                      dem=False,
                                      resampling=Resampling.bilinear):
        """Stack multiple raster band in one raster, with a specific output format and resolution
        and output bounds. It can handle the DEM = DSM - DTM computation if the dict of raster
        band includes a band called "DTM" and a band called "DSM", but you must set the dem parameter
        to True.

        Parameters
        ----------
        dict_of_raster : dict
            a dictionnary of raster definition with the nomenclature
            "name of raster":{"path": "/path/to/my/raster", "band": an array of index band of interest
            in rasterio/gdal format (starting to 1)}
        meta : dict
            a dictionary of metadata in rasterio Dataset format
        bounds : Union[Tuple, List]
            bounds delimiting the output extent
        width : int
            the width of the output
        height : int
            the height of the output
        resolution: obj:`list` of :obj: `float`
            the output resolution for x and y
        dem : bool, optional
            indicate if a DSM - DTM band is computed on the fly
            ("DSM" and "DTM" must be present in the dictionary of raster), by default False
        resampling: one enum from rasterio.Reampling
            resampling method to use when a resolution change is necessary.
            Default: Resampling.bilinear

        Returns
        -------
        numpy NDArray
            the stacked raster
        """
        handle_dem = False
        stacked_bands = None

        if "DSM" in dict_of_raster.keys() and "DTM" in dict_of_raster.keys() and dem is True:
            handle_dem = True

        for key, value in dict_of_raster.items():
            if (key not in ["DSM", "DTM"]) or handle_dem is False:
                src = value["connection"]
                window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src.meta["transform"])
                band_indices = value["bands"]
                img, _ = raster_to_ndarray_from_dataset(src,
                                                        width,
                                                        height,
                                                        resolution,
                                                        band_indices=band_indices,
                                                        resampling=resampling,
                                                        window=window)

                # pixels are normalized to [0, 1]
                img = img_as_float(img)

                LOGGER.debug(f"type of img: {type(img)}, shape {img.shape}")

                stacked_bands = img if stacked_bands is None else np.dstack([stacked_bands, img])
                LOGGER.debug(f"type of stacked bands: {type(stacked_bands)}, shape {stacked_bands.shape}")

        if handle_dem:
            dsm_ds = dict_of_raster["DSM"]["connection"]
            dtm_ds = dict_of_raster["DTM"]["connection"]
            dsm_window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], dsm_ds.meta["transform"])
            dtm_window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], dtm_ds.meta["transform"])
            band_indices = dict_of_raster["DSM"]["bands"]

            dsm_img, _ = raster_to_ndarray_from_dataset(dsm_ds,
                                                        width,
                                                        height,
                                                        resolution,
                                                        band_indices=band_indices,
                                                        resampling=resampling,
                                                        window=dsm_window)

            dtm_img, _ = raster_to_ndarray_from_dataset(dtm_ds,
                                                        width,
                                                        height,
                                                        resolution,
                                                        band_indices=band_indices,
                                                        resampling=resampling,
                                                        window=dtm_window)

            # raw dem = dsm - dtm
            img = dsm_img - dtm_img
            # LOGGER.debug(img.sum())
            # dsm should not be under dtm theorically but this could happen
            # due to product specification (rounding) etc.. so add check
            mne_msk = dtm_img > dsm_img
            img[mne_msk] = 0
            # scaling to vertical resolution such that it should be ok when convert
            # to uint8
            img *= 5  # empircally chosen factor

            # probably a wrong thing to do.. resolution[0] and 1 are x and y resolution
            # and not min/max elevation
            # xmin, xmax = max(resolution[0], resolution[1]), 255
            # img[img < xmin] = xmin  # low pass filter
            # img[img > xmax] = xmax  # high pass filter

            # low pass filer, should not do anatyhing as we should
            # already have only positives values.
            img[img < 0] = 0
            # high pass filter.
            img[img > 255] = 255

            # normalize to [0, 1]. as input are float img_as_float could not be used for
            # normalization.
            img = img / 255
            # LOGGER.debug(f"img min: {img.min()}, img max: {img.max()}, img shape: {img.shape}")
            stacked_bands = img if stacked_bands is None else np.dstack([stacked_bands, img])
            LOGGER.debug(f"type of stacked bands: {type(stacked_bands)}, shape {stacked_bands.shape}")

        return stacked_bands

    @staticmethod
    def stack_window_raster(center,
                            dict_of_raster,
                            meta,
                            dem,
                            compute_only_masks=False,
                            raster_out=None,
                            meta_msk=None,
                            output_type="uint8"):
        """stack band at window level of geotif layer and create a
        couple patch image and patch mask

        Parameters
        ----------
        center : pandas.core.series.Series
            a row from a pandas DataFrame
        dict_of_raster : dict[str, str]
            dictionary of layer geo tif (RGB, CIR, etc.)
        meta : dict
            metadata for the window raster
        dem : boolean
            rather calculate or not DMS - DMT and create a new band with it
        compute_only_masks : int (0,1)
            rather compute only masks or not
        raster_out: str
            path of rasterized full mask where to extract the window mask
        meta_msk: dict
            metadata in rasterio format for raster mask
        output_type: str
            output type of output patch raster in numpy dtype format

        Returns
        -------
            None
        """
        resampling = Resampling.bilinear

        bounds = get_bounds(center.x,
                            center.y,
                            meta['width'],
                            meta['height'],
                            meta['resolution'][0],
                            meta['resolution'][1])

        window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], meta['transform'])

        # Adapt meta for the patch
        meta_img_patch = meta.copy()
        meta_img_patch['transform'] = transform(window, meta['transform'])

        meta_msk_patch = meta_msk.copy()
        meta_msk_patch["transform"] = transform(window, meta_msk['transform'])

        # Creation of masks tiles from an input window.
        create_patch_from_center(center["msk_file"],
                                 raster_out,
                                 meta_msk_patch,
                                 window,
                                 resampling)

        if not compute_only_masks:
            img = CollectionDatasetReader.get_stacked_window_collection(dict_of_raster,
                                                                        bounds,
                                                                        meta_img_patch['width'],
                                                                        meta_img_patch['height'],
                                                                        meta_img_patch['resolution'],
                                                                        dem,
                                                                        resampling=resampling)
            raster_img = (reshape_as_raster(img) * 255).astype(output_type)

            with rasterio.open(center["img_file"], 'w', **meta_img_patch) as dst:
                dst.write(raster_img)
