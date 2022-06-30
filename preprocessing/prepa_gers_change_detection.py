import os
import sys
import glob
import shutil
import re
import glob
import json
from typing import Tuple, Union, Optional, List, Callable, Dict
from collections import OrderedDict
import numpy as np
import cv2 as cv
import geopandas as gpd
import pandas
import pandas as pd
import rasterio as rio
from rasterio.windows import from_bounds, transform
from rasterio import features
from rasterio.plot import reshape_as_raster, reshape_as_image
import fiona
from shapely.geometry import shape, box
from tqdm import tqdm
from pathlib import Path
from subprocess import PIPE, run
from rasterio.enums import Resampling
from shutil import copy2
from sklearn.model_selection import StratifiedKFold
from shapely.geometry import shape
from typing import Tuple
import rasterio as rio
from rasterio.plot import reshape_as_image, reshape_as_raster
from skimage.exposure import match_histograms
from skimage import data, segmentation
from skimage.future import graph
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from fast_slic import Slic
import cv2

SERVER_PATH = "/media/DATA10T_3/gers"
RVB_PATH_T0 = os.path.join(SERVER_PATH, "rvb-2016.vrt")
RVB_PATH_T1 = os.path.join(SERVER_PATH, "rvb-2019.vrt")
IRC_PATH_T0 = os.path.join(SERVER_PATH, "irc-2016.vrt")
IRC_PATH_T1 = os.path.join(SERVER_PATH, "irc-2019.vrt")
MNS_PATH_T0 = os.path.join(SERVER_PATH, 'mns_2016.vrt')
MNS_PATH_T1 = os.path.join(SERVER_PATH, 'mns_2019.vrt')
CONN = dict()
NPDTYPE_TO_OPENCV_DTYPE = {
    np.uint8: cv2.CV_8U,
    np.uint16: cv2.CV_16U,
    np.int32: cv2.CV_32S,
    np.float32: cv2.CV_32F,
    np.float64: cv2.CV_64F,
    np.dtype("uint8"): cv2.CV_8U,
    np.dtype("uint16"): cv2.CV_16U,
    np.dtype("int32"): cv2.CV_32S,
    np.dtype("float32"): cv2.CV_32F,
    np.dtype("float64"): cv2.CV_64F,
}
kernel20 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))
kernel10 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))
kernel5 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))


def open_connections():

    CONN['IRC_T0'] = rio.open(IRC_PATH_T0)
    CONN['IRC_T1'] = rio.open(IRC_PATH_T1)
    CONN['RVB_T0'] = rio.open(RVB_PATH_T0)
    CONN['RVB_T1'] = rio.open(RVB_PATH_T1)
    CONN['MNS_T0'] = rio.open(MNS_PATH_T0)
    CONN['MNS_T1'] = rio.open(MNS_PATH_T1)


def close_connections():
    [v.close() for v in CONN.values()]


def apply_histogram(img: np.ndarray, reference_image: np.ndarray, blend_ratio: float = 0.5) -> np.ndarray:
    if img.dtype != reference_image.dtype:
        raise RuntimeError(
            f"Dtype of image and reference image must be the same. Got {img.dtype} and {reference_image.dtype}"
        )
    reference_image = cv2.resize(reference_image, dsize=(img.shape[1], img.shape[0]))
    matched = match_histograms(np.squeeze(img), np.squeeze(reference_image), multichannel=True)
    img = cv2.addWeighted(
        matched,
        blend_ratio,
        img,
        1 - blend_ratio,
        0,
        dtype=NPDTYPE_TO_OPENCV_DTYPE[img.dtype],
    )
    return img


def difference_intensity(img_T0: np.ndarray,
                         img_T1: np.ndarray,
                         apply_hist: bool = False) -> np.ndarray:
    """

    Parameters
    ----------
    img_T0
    img_T1

    Returns
    -------

    """

    img_T0 = apply_histogram(img_T0, img_T1) if apply_hist else img_T0
    # img_T0_matched = img_T0
    distance: np.ndarray = np.sqrt(np.power(img_T0 - img_T1, 2))
    distance_min: np.ndarray = distance.min(axis=(1, 2), keepdims=True)
    distance_max: np.ndarray = distance.max(axis=(1, 2), keepdims=True)
    DI: np.ndarray = (distance - distance_min) / (distance_max - distance_min)
    return DI


def extract_bands(bounds: List) -> Tuple:

    # print(window)
    # IRC T0
    src = CONN["IRC_T0"]
    meta = src.meta
    window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], meta["transform"])
    patch_transform = transform(window, meta["transform"])
    irc_T0 = reshape_as_image(src.read(window=window))

    # IRC T1
    src = CONN["IRC_T1"]
    meta = src.meta
    window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], meta["transform"])
    patch_transform = transform(window, meta["transform"])
    irc_T1 = reshape_as_image(src.read(window=window))
    return irc_T0, irc_T1, src.profile, patch_transform


def extract_bands_mns(bounds: List) -> Tuple:

    # print(window)
    # IRC T0
    src = CONN["MNS_T0"]
    meta = src.meta
    window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], meta["transform"])
    patch_transform = transform(window, meta["transform"])
    mns_T0 = reshape_as_image(src.read(window=window))

    # IRC T1
    src = CONN["MNS_T1"]
    meta = src.meta
    window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], meta["transform"])
    patch_transform = transform(window, meta["transform"])
    mns_T1 = reshape_as_image(src.read(window=window))
    return mns_T0, mns_T1, src.profile, patch_transform


def diff_mns(mns_T0, mns_T1):

    distance: np.ndarray = np.abs(mns_T0 - mns_T1)
    distance[distance > 255.0] = 255.0
    distance[distance < 5.0] = 0.0
    return distance.astype("uint8")


def build_visual_helpers(zones: str, output_path: str):

    zones_gdf = gpd.read_file(zones)
    open_connections()
    for idx, row in tqdm(zones_gdf.iterrows()):

        irc_T0, irc_T1, profile, patch_transform = extract_bands(row.geometry.bounds)
        DI = difference_intensity(irc_T0, irc_T1, apply_hist=True)
        DI = DI.max(axis=2)
        height, width = DI.shape
        DI = np.expand_dims(DI, axis=0).astype("uint8")
        print(DI.shape)
        output_file = os.path.join(output_path, row['id_zone'] + '-DI.tif')
        profile["transform"] = patch_transform
        profile["count"] = 1
        profile['driver'] = "GTiff"
        profile['width'] = width
        profile['height'] = height
        print(profile)
        with rio.open(output_file, 'w+', **profile) as dst:
            dst.write(DI)

    close_connections()


def build_visual_helpers_mns(zones: str, output_path: str):

    zones_gdf = gpd.read_file(zones)
    open_connections()
    for idx, row in tqdm(zones_gdf.iterrows()):

        mns_T0, mns_T1, profile, patch_transform = extract_bands_mns(row.geometry.bounds)
        diff = diff_mns(mns_T0, mns_T1)
        diff = reshape_as_raster(diff)
        channel, height, width = diff.shape

        # DI = np.expand_dims(DI, axis=0).astype("uint8")
        print(diff.shape)
        output_file = os.path.join(output_path, row['id_zone'] + '-diff_mns.tif')
        output_file_t0 = os.path.join(output_path, row['id_zone'] + '-mns_T0.tif')
        output_file_T1 = os.path.join(output_path, row['id_zone'] + '-mns_T1.tif')
        profile["transform"] = patch_transform
        profile["count"] = channel
        profile['driver'] = "GTiff"
        profile['width'] = width
        profile['height'] = height
        profile['dtype'] = rio.uint8
        print(profile)
        with rio.open(output_file, 'w+', **profile) as dst:
            dst.write(diff)
        profile['dtype'] = rio.float32
        with rio.open(output_file_t0, 'w+', **profile) as dst:
            dst.write(reshape_as_raster(mns_T0))
        with rio.open(output_file_T1, 'w+', **profile) as dst:
            dst.write(reshape_as_raster(mns_T1))
    close_connections()


if __name__ == "__main__":

    """
    zones = "/home/ign.fr/skhelifi/data/gers/chantier_pilote_annotations_changement/zones_sur_annotations_segmentation/zones_vt_diff_D032_2016_2019_preremplies.shp"
    output_path = "/home/ign.fr/skhelifi/data/gers/chantier_pilote_annotations_changement/DI"
    build_visual_helpers(zones, output_path)
    """

    zones = "/home/ign.fr/skhelifi/data/gers/chantier_pilote_annotations_changement/zones_sur_annotations_segmentation/zones_vt_diff_D032_2016_2019_preremplies.shp"
    output_path = "/home/ign.fr/skhelifi/data/gers/chantier_pilote_annotations_changement/DIFF_MNS"
    build_visual_helpers_mns(zones, output_path)
