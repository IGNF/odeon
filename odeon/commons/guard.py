"""The Guardian Angel Of Odeon
Module to place every control checking security function or class

"""
import rasterio
import fiona
from odeon.commons.exception import ErrorCodes, OdeonError
from fiona import supported_drivers
import os

RASTER_DRIVER_ACCEPTED = ["GTiff", "GeoTIFF"]
FIONA_DRIVER_ACCEPTED = supported_drivers.keys()


def geo_projection_raster_guard(raster):
    """

    Parameters
    ----------
    raster : dict
     file path of raster

    Returns
    -------
    None

    Raises
    -------
    odeon.commons.exception.OdeonError
     error code ERR_COORDINATE_REFERENCE_SYSTEM
    """
    try:

        with rasterio.open(raster) as src:

            if src.crs == "" or src.crs is None:

                raise OdeonError(ErrorCodes.ERR_COORDINATE_REFERENCE_SYSTEM,
                                 f"the crs {src.crs} of raster {raster} is empty")

    except rasterio.errors.RasterioError as rioe:

        raise OdeonError(ErrorCodes.ERR_IO,
                         f"the crs {src.crs} of raster {raster} is empty", stack_trace=rioe)

    except rasterio.errors.RasterioIOError as rioe:

        raise OdeonError(ErrorCodes.ERR_IO,
                         f"the crs {src.crs} of raster {raster} is empty", stack_trace=rioe)


def geo_projection_vector_guard(vector):
    """

    Parameters
    ----------
    vector : str
     file path of vector

    Returns
    -------
    None

    Raises
    -------
    odeon.commons.exception.OdeonError
     error code ERR_COORDINATE_REFERENCE_SYSTEM
    """

    with fiona.open(vector) as src:

        if src.crs == "" or src.crs is None:

            raise OdeonError(ErrorCodes.ERR_COORDINATE_REFERENCE_SYSTEM,
                             f"the crs {src.crs} of vector {vector} is empty")


def vector_driver_guard(vector):
    """

    Parameters
    ----------
    vector : str
     path of vector file

    Returns
    -------

    Raises
    -------
    odeon.commons.exception.OdeonError
     error code ERR_DRIVER_COMPATIBILITY
    """

    with fiona.open(vector) as src:

        if src.driver not in FIONA_DRIVER_ACCEPTED:

            raise OdeonError(ErrorCodes.ERR_DRIVER_COMPATIBILITY,
                             f"the driver {src.driver} of mask file"
                             f" {vector} is not accepted in Odeon")


def raster_driver_guard(raster):
    """

        Parameters
        ----------
        raster : str
         path of vector file

        Returns
        -------

        Raises
        -------
        odeon.commons.exception.OdeonError
         error code ERR_DRIVER_COMPATIBILITY
        """
    try:

        with rasterio.open(raster) as src:

            if src.driver not in RASTER_DRIVER_ACCEPTED:

                raise OdeonError(ErrorCodes.ERR_DRIVER_COMPATIBILITY,
                                 f"the driver {src.driver} of raster file"
                                 f" {raster} is not accepted in Odeon")

    except rasterio.errors.RasterioError as rioe:

        raise OdeonError(ErrorCodes.ERR_IO,
                         f"the crs {src.crs} of raster {raster} is empty", stack_trace=rioe)

    except rasterio.errors.RasterioIOError as rioe:

        raise OdeonError(ErrorCodes.ERR_IO,
                         f"the crs {src.crs} of raster {raster} is empty", stack_trace=rioe)


def files_exist(list_of_file):
    """

    Parameters
    ----------
    list_of_file : list

    Returns
    -------

    Raises
    -------
    odeon.commons.exception.OdeonError
     error code ERR_FILE_NOT_EXIST

    """

    for file in list_of_file:

        if os.path.isfile(file) is not True:

            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"the file {file} doesn't exists")


def dirs_exist(list_of_dir):
    """

    Parameters
    ----------
    list_of_dir : list

    Returns
    -------

    Raises
    -------
    odeon.commons.exception.OdeonError
     error code ERR_DIR_NOT_EXIST

    """

    for dir_name in list_of_dir:

        if os.path.isdir(dir_name) is not True:
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"the dir {dir_name} doesn't exists")


def raster_bands_exist(raster, list_of_band):
    """

    Parameters
    ----------
    raster
    list_of_band

    Returns
    -------

    Raises
    -------
    odeon.commons.exception.OdeonError
     error code ERR_DIR_NOT_EXIST

    """
    try:

        with rasterio.open(raster) as src:

            bands_count = src.count

            for band in list_of_band:

                if band > bands_count:
                    raise OdeonError(ErrorCodes.ERR_RASTER_BAND_NOT_EXIST,
                                     f"the band {band} from raster {raster} "
                                     f"doesn't exists")

    except rasterio.errors.RasterioError as rioe:

        raise OdeonError(ErrorCodes.ERR_IO,
                         f"the crs {src.crs} of raster {raster} is empty", stack_trace=rioe)

    except rasterio.errors.RasterioIOError as rioe:

        raise OdeonError(ErrorCodes.ERR_IO,
                         f"the crs {src.crs} of raster {raster} is empty", stack_trace=rioe)
