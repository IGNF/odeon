"""The Guardian Angel Of Odeon
Module to place every control checking security function or class

"""
import rasterio
import fiona
from odeon.commons.exception import ErrorCodes, OdeonError
from fiona import supported_drivers
from typing import List
from omegaconf.listconfig import ListConfig
import os
from odeon import LOGGER

RASTER_DRIVER_ACCEPTED = ["GTiff", "GeoTIFF", "VRT"]
FIONA_DRIVER_ACCEPTED = supported_drivers.keys()


def geo_projection_raster_guard(raster):
    """

    Parameters
    ----------
    raster : Union[str, list]
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
        if isinstance(raster, str):
            with rasterio.open(raster) as src:
                if src.crs == "" or src.crs is None:
                    raise OdeonError(ErrorCodes.ERR_COORDINATE_REFERENCE_SYSTEM,
                                     f"the crs {src.crs} of raster {raster} is empty")
        else:
            for file in raster:
                with rasterio.open(file) as src:
                    if src.crs == "" or src.crs is None:
                        raise OdeonError(ErrorCodes.ERR_COORDINATE_REFERENCE_SYSTEM,
                                         f"the crs {src.crs} of raster {file} is empty")

    except rasterio.errors.RasterioError as rioe:
        raise OdeonError(ErrorCodes.ERR_IO,
                         f"Odeon encountered an error during raster opening {raster}", stack_trace=rioe)

    except rasterio.errors.RasterioIOError as rioe:
        raise OdeonError(ErrorCodes.ERR_IO,
                         f"Odeon encountered an error during raster opening {raster}", stack_trace=rioe)


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
    if isinstance(vector, str):
        LOGGER.debug(vector)
        try:
            with fiona.open(vector) as src:
                if src.crs == "" or src.crs is None:
                    raise OdeonError(ErrorCodes.ERR_COORDINATE_REFERENCE_SYSTEM,
                                     f"the crs {src.crs} of vector {vector} is empty")
        except fiona._err.CPLE_AppDefinedError as error:
            raise OdeonError(ErrorCodes.ERR_COORDINATE_REFERENCE_SYSTEM,
                             f"the crs {src.crs} of vector {vector} has an encoding problem", stack_trace=error)
    else:
        for v in vector:
            try:
                LOGGER.debug(v)
                with fiona.open(v) as src:
                    if isinstance(v, str):
                        if src.crs == "" or src.crs is None:
                            raise OdeonError(ErrorCodes.ERR_COORDINATE_REFERENCE_SYSTEM,
                                             f"the crs {src.crs} of vector {v} is empty")
            except fiona._err.CPLE_AppDefinedError as error:
                raise OdeonError(ErrorCodes.ERR_COORDINATE_REFERENCE_SYSTEM,
                                 f"the crs {src.crs} of vector {vector} has an encoding problem", stack_trace=error)


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
    if isinstance(vector, str):
        with fiona.open(vector) as src:
            if src.driver not in FIONA_DRIVER_ACCEPTED:
                raise OdeonError(ErrorCodes.ERR_DRIVER_COMPATIBILITY,
                                 f"the driver {src.driver} of mask file"
                                 f" {vector} is not accepted in Odeon")
    else:
        for v in vector:
            with fiona.open(v) as src:
                if src.driver not in FIONA_DRIVER_ACCEPTED:
                    raise OdeonError(ErrorCodes.ERR_DRIVER_COMPATIBILITY,
                                     f"the driver {src.driver} of mask file"
                                     f" {v} is not accepted in Odeon")


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
        if isinstance(raster, str):
            with rasterio.open(raster) as src:
                if src.driver not in RASTER_DRIVER_ACCEPTED:
                    raise OdeonError(ErrorCodes.ERR_DRIVER_COMPATIBILITY,
                                     f"the driver {src.driver} of raster file"
                                     f" {raster} is not accepted in Odeon")
        else:
            for r in raster:
                with rasterio.open(r) as src:
                    if src.driver not in RASTER_DRIVER_ACCEPTED:
                        raise OdeonError(ErrorCodes.ERR_DRIVER_COMPATIBILITY,
                                         f"the driver {src.driver} of raster file"
                                         f" {r} is not accepted in Odeon")
    except rasterio.errors.RasterioError as rioe:
        raise OdeonError(ErrorCodes.ERR_IO,
                         f"Odeon encountered an error during raster opening {raster}", stack_trace=rioe)
    except rasterio.errors.RasterioIOError as rioe:
        raise OdeonError(ErrorCodes.ERR_IO,
                         f"Odeon encountered an error during raster opening {raster}", stack_trace=rioe)


def file_exist(file):
    if os.path.isfile(file) is not True:
        raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                            f"the file {file} doesn't exists")


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
    for element in list_of_file:
        if isinstance(element, str):
            if os.path.isfile(element) is not True:
                raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                                 f"the file {element} doesn't exists")
        else:
            for sub_element in element:
                if os.path.isfile(sub_element) is not True:
                    raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                                     f"the file {sub_element} doesn't exists")


def check_files(list_files):
    try:
        files_exist(list_files)  
    except OdeonError as error:
        raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                         "something went wrong during training configuration",
                         stack_trace=error)


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


def is_valid_dataset_path(uri):
    """

    Parameters
    ----------
    uri : str
     a string representing

    Returns
    -------
    boolean
     True if it's an extisting dir or a file, else False
    """
    if os.path.isdir(uri):
        try:
            dirs_exist([os.path.join(uri, "img"), os.path.join(uri, "msk")])
            return True
        except OdeonError as error:
            LOGGER.error(error)
            return False
    elif os.path.isfile(uri) and uri.endswith(".csv"):
        return True
    else:
        return False


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
        if isinstance(raster, str):
            with rasterio.open(raster) as src:
                bands_count = src.count
                for band in list_of_band:
                    if band > bands_count:
                        raise OdeonError(ErrorCodes.ERR_RASTER_BAND_NOT_EXIST,
                                         f"the band {band} from raster {raster} "
                                         f"doesn't exists")
        else:
            for r in raster:
                with rasterio.open(r) as src:
                    bands_count = src.count
                    for band in list_of_band:
                        if band > bands_count:
                            raise OdeonError(ErrorCodes.ERR_RASTER_BAND_NOT_EXIST,
                                             f"the band {band} from raster {r} "
                                             f"doesn't exists")
    except rasterio.errors.RasterioError as rioe:
        raise OdeonError(ErrorCodes.ERR_IO,
                         f"the crs {src.crs} of raster {raster} is empty", stack_trace=rioe)
    except rasterio.errors.RasterioIOError as rioe:
        raise OdeonError(ErrorCodes.ERR_IO,
                         f"the crs {src.crs} of raster {raster} is empty", stack_trace=rioe)


def check_raster_bands(raster_band, proposed_bands):
    """Check if the bands in the configuration file are correct and correspond to the bands in the raster.

    Parameters
    ----------
    raster_band : list
        Bands found by opening the first sample of the dataset.
    proposed_bands : list
        Bands proposed in the configuration file.
    """
    if isinstance(proposed_bands, (List, ListConfig)) and len(proposed_bands) >= 1:
        if not all((band in raster_band for band in proposed_bands)):
            LOGGER.error('ERROR: the bands in the configuration file do not correspond\
            to the available bands in the image.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                "The input parameters mask_bands and pred_bands are incorrect.")
    else:
        LOGGER.error('ERROR: bands must be a list with a length greater than 1.')
        raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                            "The input parameters mask_bands and pred_bands are incorrect.")