"""
# Generation grid module
main entry point to generation tool


"""

import rasterio
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from odeon.commons.shape import stack_shape
from odeon.commons.rasterio import rasterize_shape
from odeon.commons.rasterio import stack_window_raster
from odeon.commons.rasterio import get_scale_factor_and_img_size, get_max_type
from odeon import LOGGER
from odeon.commons.dataframe import set_path_to_center, split_dataset_from_df
from odeon.commons.folder_manager import build_directories
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.commons.guard import geo_projection_raster_guard
from odeon.commons.guard import geo_projection_vector_guard
from odeon.commons.guard import raster_driver_guard
from odeon.commons.guard import vector_driver_guard
from odeon.commons.guard import files_exist
from odeon.commons.guard import dirs_exist
from odeon.commons.guard import raster_bands_exist
from odeon.commons.exception import OdeonError, ErrorCodes

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_generation")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)


def generate(image_layers,
             vector_classes,
             output_path=None,
             poi_pattern=None,
             train_test_split=0.8,
             train_val_split=0.8,
             compute_only_masks=False,
             mns_mnt=True,
             append=False,
             image_size_pixel=512,
             resolution=None
             ):
    """Main generation function called when you use the cli tool generation

    Parameters
    ----------
    image_layers : dict
     a dict in the form {band_name: {path: file_path,band: []}
    vector_classes : dict
     a dict in the form {class_name: file_path}
    output_path : str
     the output path of generaiton
    poi_pattern : regex
     a regex to list the files of poi in input
    train_test_split : float
     the percentage of train in train test split
    train_val_split : float
     the percentage of train in train val split
    compute_only_masks : bool
     rather generate only the mask or not
    mns_mnt : bool
     create the MNS - MNT band or not if MNS and MNT are here
    append : bool
     append generated data to a previous a generation output
    image_size_pixel : int
     size of patch in pixel
    resolution : list(float, float)
     resolution in x and y of the output patches

    Returns
    -------

    """

    """ Retrieve config variables """
    shape_files = vector_classes
    raster_out = os.path.join(output_path,
                              "full_mask.tiff")
    dict_of_raster = image_layers
    img_size = image_size_pixel
    files = glob.glob(poi_pattern)
    nb_of_raster_bands = 0

    try:

        files_exist(files)
        dirs_exist([output_path])

    except OdeonError as oe:

        raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                         "something went wrong during generation", stack_trace=oe)

    # we compute the number of bands for the patch image output
    # and the scales factors for each raster with the targeted resolution
    for raster_name, raster in dict_of_raster.items():

        try:

            files_exist([raster["path"]])
            geo_projection_raster_guard(raster["path"])
            raster_driver_guard(raster["path"])
            raster_bands_exist(raster["path"], raster["bands"])

        except OdeonError as oe:

            raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                             "something went wrong during generation",
                             stack_trace=oe)

        nb_of_raster_bands += len(raster["bands"])

        x_scale, y_scale, scaled_width, scaled_height = \
            get_scale_factor_and_img_size(raster["path"],
                                          resolution,
                                          img_size,
                                          img_size)
        raster["x_scale"] = x_scale
        raster["y_scale"] = y_scale
        raster["scaled_width"] = scaled_width
        raster["scaled_height"] = scaled_height

    if ("DSM" and "DTM") in dict_of_raster and mns_mnt:

        nb_of_raster_bands -= 1

    for name, vector in shape_files.items():
        try:

            files_exist([vector])
            geo_projection_vector_guard(vector)
            vector_driver_guard(vector)

        except OdeonError as oe:

            raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                             "something went wrong during generation",
                             stack_trace=oe)

    stdout = '''
    ##############################################
    #                                            #
    #   Preprocess: Shapefile rasterization      #
    #                                            #
    ##############################################
    '''

    STD_OUT_LOGGER.info(stdout)

    with rasterio.open(next(iter(dict_of_raster.values()))["path"]) as dataset:

        meta = dataset.meta
        meta["dtype"] = rasterio.uint8
        meta["count"] = len(shape_files) + 1
        shape = dataset.shape

    with rasterio.open(raster_out, 'w', **meta) as new_dataset:

        # other_band: mask of non classified pixel
        other_band = np.ones((meta["height"], meta["width"]), dtype=np.uint8)

        LOGGER.debug("others band {} {}".format(other_band.shape,
                                                other_band.dtype))

        for idx, (name, shape_file) in enumerate(shape_files.items(), start=1):

            LOGGER.info("rasterizing {} class shapes".format(name))
            LOGGER.debug(shape_file)
            tuples = stack_shape(shape_file, value=1)
            band = rasterize_shape(tuples, meta, shape)
            LOGGER.debug(band.shape)
            other_band = other_band * (band == 0).astype(np.uint8)
            new_dataset.write_band(idx, band)

        LOGGER.debug(np.sum(other_band == 0))
        new_dataset.write_band(len(shape_files) + 1, other_band)

    stdout = '''
    ##############################################
    #                                            #
    #   Prepraring directories                   #
    #                                            #
    ##############################################
    '''

    STD_OUT_LOGGER.info(stdout)

    train_path = os.path.join(output_path, "train")
    test_path = os.path.join(output_path, "test")
    val_path = os.path.join(output_path, "val")
    train_img_path = os.path.join(train_path, "img")
    train_msk_path = os.path.join(train_path, "msk")
    test_img_path = os.path.join(test_path, "img")
    test_msk_path = os.path.join(test_path, "msk")
    val_img_path = os.path.join(val_path, "img")
    val_msk_path = os.path.join(val_path, "msk")

    paths = {
        "train_img_path": train_img_path,
        "train_msk_path": train_msk_path,
        "test_img_path": test_img_path,
        "test_msk_path": test_msk_path,
        "val_img_path": val_img_path,
        "val_msk_path": val_msk_path
    }

    build_directories(paths, append)

    stdout = '''
    ##############################################
    #                                            #
    #   Generating images and mask               #
    #                                            #
    ##############################################
    '''
    STD_OUT_LOGGER.info(stdout)

    """ we preapare the metadata for the img tiles and the mask tiles"""

    with rasterio.open(next(iter(dict_of_raster.values()))["path"]) as dst:

        meta_msk = dst.meta.copy()
        meta_msk["count"] = len(vector_classes) + 1
        meta_msk["width"] = img_size
        meta_msk["height"] = img_size
        meta_img = meta_msk.copy()
        meta_msk["dtype"] = rasterio.uint8
        meta_img["dtype"] = get_max_type(dict_of_raster)

    meta_img["count"] = nb_of_raster_bands

    """ iterates over csv files """
    for file in tqdm(files):

        centers = pd.read_csv(file, delimiter=";",
                              header=None, names=["x", "y"])

        centers["img_file"] = ""
        centers["msk_file"] = ""

        train_val, test = split_dataset_from_df(centers,
                                                train_test_split)
        train, val = split_dataset_from_df(train_val,
                                           train_val_split)
        train = set_path_to_center(train,
                                   train_img_path,
                                   train_msk_path)
        test = set_path_to_center(test,
                                  test_img_path,
                                  test_msk_path)
        val = set_path_to_center(val,
                                 val_img_path,
                                 val_msk_path)

        LOGGER.info("generating training data,"
                    " validation and test will follow...")
        generate_data(train,
                      meta_msk,
                      meta_img,
                      raster_out,
                      dict_of_raster,
                      mns_mnt,
                      compute_only_masks)

        LOGGER.info("generating validation data,"
                    " validation and testing data will follow...")
        generate_data(val,
                      meta_msk,
                      meta_img,
                      raster_out,
                      dict_of_raster,
                      mns_mnt,
                      compute_only_masks)

        LOGGER.info("generating test data, you're almost there!!!!")
        generate_data(test,
                      meta_msk,
                      meta_img,
                      raster_out,
                      dict_of_raster,
                      mns_mnt,
                      compute_only_masks)

        train[["img_file", "msk_file"]].to_csv(os.path.join(output_path,
                                                            "train.csv"),
                                               index=False,
                                               header=False)
        val[["img_file", "msk_file"]].to_csv(os.path.join(output_path,
                                                          "val.csv"),
                                             index=False,
                                             header=False)
        test[["img_file", "msk_file"]].to_csv(os.path.join(output_path,
                                                           "test.csv"),
                                              index=False,
                                              header=False)


def generate_data(df,
                  meta_msk,
                  meta_img,
                  raster_out,
                  dict_of_raster,
                  mns_mnt,
                  compute_only_masks):
    """
    generate and writes a Patch (a tuple image, mask) for each
    row of the dataframe representing the Dataset.

    Parameters
    ----------
    df : pandas.DataFrame
     the list of center point for each patch
    meta_msk : dict
     rasterio metadata for mask generation
    meta_img : dict
     rasterio metadata for img generation
    raster_out : str
     path to write
    dict_of_raster : dict
     a dictionary of raster name, raster file
    mns_mnt : bool
     rather calculate or not DMS - DMT and create a new band with it
    compute_only_masks : bool
     computer only mask or not


    Returns
    -------
    None

    """

    for idx, center in tqdm(df.iterrows(), total=len(df)):

        try:

            stack_window_raster(center,
                                dict_of_raster,
                                meta_img,
                                mns_mnt,
                                compute_only_masks,
                                raster_out,
                                meta_msk)

        except Exception as error:

            raise OdeonError(ErrorCodes.ERR_GENERATION_ERROR,
                             "something went wrong during generation",
                             stack_trace=error)
