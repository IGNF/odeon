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
import cv2
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

tests_zones = ["v15", "v5", "v11", "u4", "u8", "u10", "n2", "n14", "n18"]
# ROOT = "/media"
ROOT = "/media/HP-2007S005-media"
# ROOT_OUT = "/var/data/dl"
ROOT_OUT = "/media/HP-2007S005-data"
# NAS = "/media/NAS"
NAS = "/media/HP-2007S005-media/NAS"
LIVRAISON_2016_PATH = "smb://store-sbv/pbf/TERR-IA/ANNOTATIONS/CHANTIER_GERS_2016/C_RETOUR"
LIVRAISON_2016_PATTERN = "*/projet_qgs_ini/FINAL_DATA/*Final.shp"
GERS_DATA_PATH = os.path.join(ROOT_OUT, "gers")
NOMENCLATURE_PATH = os.path.join(ROOT, "NAS/OCSNG_GERS_2021/nomenclature")
OLD_NOMENCLATURE_PATH = os.path.join(ROOT, "NAS/OCSNG_GERS_2021/nomenclature_old")
PATH_TO_2019 = os.path.join(ROOT, "NAS/OCSNG_GERS_2021/annotation/dataset_32_2019")
PATH_TO_2016 = os.path.join(ROOT, "NAS/OCSNG_GERS_2021/annotation/dataset_32_2016")
RVB_FILE = "*rvb.tif"
IRC_FILE = "*irc.tif"
IRC_FDM_FILE = "*irc_with_*_radiometry_with_tool_fdm.tif"
RVB_FDM_FILE = "*rvb_with_*_radiometry_with_tool_fdm.tif"
IRC_HM_FILE = "*irc_with_*_radiometry_with_tool_hm.tif"
RVB_HM_FILE = "*rvb_with_*_radiometry_with_tool_hm.tif"
IRC_MM_FILE = "*irc_with_*_radiometry_with_tool_mm.tif"
RVB_MM_FILE = "*rvb_with_*_radiometry_with_tool_mm.tif"
MNH_FILE = "*mnh.tif"
GT_FILE = "*saisie.shp"
GT_PATTERN = "*saisie.*"

nomenclature_v1 = {
'coupe': 'broussaille',
'vigne': 'broussaille',
'terre_labouree': 'culture',
'pelouse': 'culture',
}
def tile_dep(gdf_file, dep, zones_path="/media/HP-2007S005-media/NAS/OCSNG_GERS_2021/annotation/dataset_32_2019", tile_size=256):
    """Build a tiling for a department based on its extent
    and a size of tile.

    Parameters
    ----------
    gdf_file a shape file of France department extent
    dep the code of the department of interest
    tile_size the size of tiling in unit CRS system (meter, degree)

    Returns
    -------
    None
    """
    print(gdf_file)
    gdf = gpd.read_file(gdf_file)
    # zones = glob.glob(os.path.join(zones_path, "*/*mask.gpkg"))
    # gdf_list = [gpd.read_file(f) for f in zones]
    # zones_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    # zones_gdf.to_file("/media/HP-2007S005-media/NAS/OCSNG_GERS_2021/annotation/dataset_32_2019/zones_gdf.geojson", driver="GeoJSON", crs=gdf.crs)
    zones_gdf = gpd.read_file("/media/HP-2007S005-media/NAS/OCSNG_GERS_2021/annotation/dataset_32_2019/zones_gdf.geojson")
    print(zones_gdf)
    output_file_unsupervised = os.path.join('/media/HP-2007S005-data/gers', str(dep) + '_' + str(tile_size) + 'X' + str(tile_size) + '_unsupervised.geojson')
    print(output_file_unsupervised)
    print(gdf.columns)
    dep_gdf = gpd.GeoDataFrame(gdf[gdf["code_dep"] == str(dep)], crs=gdf.crs)
    bounds = dep_gdf.iloc[0].geometry.bounds
    x_range = np.arange(int(bounds[0]), int(bounds[2]), tile_size)
    y_range = np.arange(int(bounds[1]), int(bounds[3]), tile_size)
    tmp = list()
    idx = 0

    for i in tqdm(x_range):

        for j in y_range:

            bbox = box(i, j, i + tile_size, j + tile_size)
            tmp.append({"id_box": idx, "dep": dep, "geometry": bbox})
            idx += 1

    output_gdf = gpd.GeoDataFrame(tmp, crs=gdf.crs)
    print(len(output_gdf))
    joined = gpd.sjoin(output_gdf, dep_gdf, how="inner", op="within", rsuffix="r")
    print(len(joined))
    output_gdf = output_gdf[output_gdf["id_box"].isin(joined["id_box"].unique())]
    print(len(output_gdf))
    output_gdf = gpd.GeoDataFrame(output_gdf, geometry="geometry", crs=gdf.crs)
    joined = gpd.sjoin(output_gdf, zones_gdf, how="inner", op="intersects", rsuffix="r")
    print(len(joined))
    print(joined.columns)
    print(output_gdf.columns)
    output_gdf = output_gdf[~output_gdf["id_box"].isin(joined["id_box"].unique())]
    output_gdf = output_gdf.sample(len(joined) * 6)
    print(len(output_gdf))
    exit(0)
    output_gdf.to_file(output_file_unsupervised, driver="GeoJSON")


def attach_zone_to_patch(gdf_patch_file, gdf_zone_file, output_file):

    gdf_patch = gpd.read_file(gdf_patch_file)
    gdf_zone = gpd.read_file(gdf_zone_file)
    join_gdf = gpd.sjoin(gdf_patch, gdf_zone, how="left", op="within")
    print(len(gdf_patch))
    print(len(join_gdf))
    print(join_gdf)
    join_gdf["test"] = join_gdf["id"].apply(lambda x: x in tests_zones)
    join_gdf.to_file(output_file, crs=gdf_zone.crs, driver="GeoJSON")


def kfold_split(gdf_file, output_path, n_splits=3):

    gdf = gpd.read_file(gdf_file)
    gdf[["fold_1_train", "fold_1_val", "fold_2_train", "fold_2_val", "fold_3_train", "fold_3_val"]] = [0, 0, 0, 0, 0, 0]
    zone_type = {"n": 0, "v": 1, "u": 2}
    gdf["zone_type"] = gdf["id"].apply(lambda x: zone_type[x[0]])
    print(len(gdf))
    train_gdf = gdf[~gdf["id"].isin(tests_zones)]
    print(len(train_gdf))
    skf = StratifiedKFold(n_splits=n_splits)
    X = train_gdf.index
    y = train_gdf.zone_type
    fold_cols = ["fold_1_train", "fold_1_val", "fold_2_train", "fold_2_val", "fold_3_train", "fold_3_val"]
    gdf[fold_cols] = 0
    split = 1

    for train_index, val_index in skf.split(X, y):

        gdf.loc[train_index, f"fold_{split}_train"] = 1
        gdf.loc[val_index, f"fold_{split}_val"] = 1
        split += 1

    print(gdf)
    gdf.to_file(output_path, crs=gdf.crs, driver="GeoJSON")


def generate_dataset(gdf_file, zone_file):

    patch_gdf = gpd.read_file(gdf_file)
    print(patch_gdf)
    zone_gdf = gpd.read_file(zone_file)
    print(zone_gdf.columns)
    zone_gdf.drop("2016_updat", axis=1, inplace=True)
    print(zone_gdf)
    joined_gdf = gpd.sjoin(patch_gdf, zone_gdf, how="left", op="within")
    # build dict of data
    out_gdf = gpd.GeoDataFrame(joined_gdf, crs=patch_gdf.crs, geometry="geometry")
    out_gdf.to_file(os.path.join(GERS_DATA_PATH, "supervised_dataset_db_new.geojson"), driver="GeoJSON")


def build_nomenclature_db():

    # saisie
    saisie_df = pd.read_csv(os.path.join(NOMENCLATURE_PATH, "saisie.csv"))
    saisie_df = saisie_df.rename(columns={"legende ": "legende"})
    saisie_df = saisie_df.astype({"saisie": np.uint8, "legende": str})
    print(saisie_df.columns)
    saisie_clut_df = pd.read_csv(os.path.join(OLD_NOMENCLATURE_PATH, "clut_saisie.txt"),
                                 delim_whitespace=True,
                                 names=["saisie", "R_saisie", "G_saisie", "B_saisie"])
    saisie_clut_df = saisie_clut_df.astype({"saisie": np.uint8,
                                            "R_saisie": str,
                                            "G_saisie": str,
                                            "B_saisie": str})

    print(saisie_clut_df)

    # NAF
    naf_df = pd.read_csv(os.path.join(NOMENCLATURE_PATH, "naf.csv"))
    naf_df = naf_df.rename(columns={"legende ": "legende_naf"})
    naf_saisie_df = pd.read_csv(os.path.join(NOMENCLATURE_PATH, "saisie_naf.csv"))
    naf_saisie_df = naf_saisie_df.astype({"saisie": str, "naf": str})
    naf_clut_df = pd.read_csv(os.path.join(NOMENCLATURE_PATH, "clut_naf.txt"),
                              delim_whitespace=True,
                              names=["naf", "R_naf", "G_naf", "B_naf"])
    naf_clut_df = naf_clut_df.astype({"naf": str, "R_naf": str, "G_naf": str, "B_naf": str})
    naf_df["naf"] = naf_df["naf"].astype(np.uint8)
    naf_saisie_df["naf"] = naf_saisie_df["naf"].astype(np.uint8)
    naf_df = pd.merge(naf_df, naf_saisie_df, left_on="naf", right_on="naf")
    print(naf_df)
    print(naf_saisie_df)
    naf_df["naf"] = naf_df["naf"].astype(np.uint8)
    naf_clut_df["naf"] = naf_clut_df["naf"].astype(np.uint8)
    naf_df = pd.merge(naf_df, naf_clut_df, left_on="naf", right_on="naf")
    naf_df["naf"] = naf_df["naf"].astype(np.uint8)
    print(naf_df)
    print(naf_df.dtypes)
    saisie_df["saisie"] = saisie_df["saisie"].astype(np.uint8)
    saisie_clut_df["saisie"] = saisie_clut_df["saisie"].astype(np.uint8)
    saisie_df = pd.merge(saisie_df, saisie_clut_df, left_on="saisie", right_on="saisie")
    print(saisie_df)
    saisie_df["saisie"] = saisie_df["saisie"].astype(np.uint8)
    naf_df["saisie"] = naf_df["saisie"].astype(np.uint8)
    saisie_df = pd.merge(saisie_df, naf_df, left_on="saisie", right_on="saisie")

    # URBAIN
    urbain_df = pd.read_csv(os.path.join(OLD_NOMENCLATURE_PATH, "urbain.csv"))
    urbain_df = urbain_df.rename(columns={"saisie": "urbain", "legende ": "legende_urbain"})
    urbain_saisie_df = pd.read_csv(os.path.join(OLD_NOMENCLATURE_PATH, "saisie_urbain.csv"))
    urbain_saisie_df = urbain_saisie_df.astype({"saisie": str, "urbain": str})
    urbain_clut_df = pd.read_csv(os.path.join(OLD_NOMENCLATURE_PATH, "clut_urbain.txt"),
                                              delim_whitespace=True,
                                              names=["urbain", "R_urbain", "G_urbain", "B_urbain"])
    urbain_clut_df = urbain_clut_df.astype({"urbain": str, "R_urbain": str, "G_urbain": str, "B_urbain": str})
    urbain_df["urbain"] = urbain_df["urbain"].astype(np.uint8)
    urbain_saisie_df["urbain"] = urbain_saisie_df["urbain"].astype(np.uint8)
    urbain_df = pd.merge(urbain_df, urbain_saisie_df, left_on="urbain", right_on="urbain")
    print(urbain_df)
    print(urbain_saisie_df)
    urbain_df["urbain"] = urbain_df["urbain"].astype(np.uint8)
    urbain_clut_df["urbain"] = urbain_clut_df["urbain"].astype(np.uint8)
    urbain_df = pd.merge(urbain_df, urbain_clut_df, left_on="urbain", right_on="urbain")
    urbain_df["urbain"] = urbain_df["urbain"].astype(np.uint8)
    print(urbain_df)
    print(urbain_df.dtypes)
    saisie_df["saisie"] = saisie_df["saisie"].astype(np.uint8)
    saisie_clut_df["saisie"] = saisie_clut_df["saisie"].astype(np.uint8)
    saisie_df = pd.merge(saisie_df, saisie_clut_df, left_on="saisie", right_on="saisie")
    print(saisie_df)
    saisie_df["saisie"] = saisie_df["saisie"].astype(np.uint8)
    urbain_df["saisie"] = urbain_df["saisie"].astype(np.uint8)
    saisie_df = pd.merge(saisie_df, urbain_df, left_on="saisie", right_on="saisie")
    saisie_df = saisie_df.rename(columns={"R_saisie_x": "R_saisie", "G_saisie_x": "G_saisie", "B_saisie_x": "B_saisie"})
    saisie_df = saisie_df.drop(["R_saisie_y", "G_saisie_y", "B_saisie_y"], axis=1)
    print(saisie_df)
    saisie_df.to_csv(os.path.join(GERS_DATA_PATH, "nomenclature.csv"))


def get_meta(f):

    with rio.open(f) as src:
        meta = src.meta
        return meta


def get_patch_from_raster(f, window):

    with rio.open(f) as src:
        bands = src.read(window=window)
        return bands


def make_chips(db_file, output_path):

    # nomenclature_df = pd.read_csv(nomenclature_file)
    # channel_raster = ["R", "V", "B", "IR", "MNH", "R-HM", "G-HM", "B-HM"]
    # histo_raster_2019 = {channel: None for channel in channel_raster}
    # histo_raster_2016 = {channel: None for channel in channel_raster}

    db = gpd.read_file(db_file)
    dtype_out = "uint8"
    counts_raster_out = 8
    counts_gt_out = 1

    print(output_path)
    output_path_2019 = Path(os.path.join(output_path, "2019")).mkdir(parents=True, exist_ok=True)
    print(str(output_path_2019))
    grouped_db = db.groupby(["id"])

    for zone_idx, zone_gdf in tqdm(grouped_db, total=len(grouped_db)):

        # get connections
        connections = {
            "rvb_2019": {"path": os.path.join(PATH_TO_2019, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_rvb.tif"])},
            "irc_2019": {"path": os.path.join(PATH_TO_2019, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_irc.tif"])},
            "mnh_2019": {"path": os.path.join(PATH_TO_2019, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_mnh.tif"])},
            "rvb_hm_2019": {"path": os.path.join(PATH_TO_2019, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_rvb_with_2016_radiometry_with_tool_hm.tif"])},
            "gt_naf_2019": {"path": os.path.join(PATH_TO_2019, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_annotation_naf.tif"])},
            "gt_urbain_2019": {"path": os.path.join(PATH_TO_2019, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_annotation_urbain.tif"])}
        }

        updated = zone_gdf["2016_updated"].iloc[0]
        is_test = zone_gdf["test"].iloc[0]
        relative_output_path_zone = "test" if is_test else "train"
        output_path_zone = os.path.join(output_path, relative_output_path_zone)
        print(f"updated {updated}")
        print(f"test {is_test}")

        if updated:

            connections.update(
                {
                "rvb_2016": {"path": os.path.join(PATH_TO_2016, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_rvb.tif"])},
                "irc_2016": {"path": os.path.join(PATH_TO_2016, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_irc.tif"])},
                "mnh_2016": {"path": os.path.join(PATH_TO_2016, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_mnh.tif"])},
                "rvb_hm_2016": {"path": os.path.join(PATH_TO_2016, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_rvb_with_2019_radiometry_with_tool_hm.tif"])},
                "gt_naf_2016": {"path": os.path.join(PATH_TO_2016, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_annotation_naf_updated.tif"])},
                "gt_urbain_2016": {"path": os.path.join(PATH_TO_2016, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_annotation_urbain_updated.tif"])},
                "gt_naf_change": {"path": os.path.join(PATH_TO_2016, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_annotation_naf_change.tif"])},
                "gt_urbain_change": {"path": os.path.join(PATH_TO_2016, *[f"zone_{str(zone_idx)}", f"zone_{str(zone_idx)}_annotation_urbain_change.tif"])}
                }
            )
        meta = None

        for k, v in connections.items():

            connections[k]["conn"] = rio.open(v["path"])
            if meta is None:
                meta = connections[k]["conn"].meta.copy()

        meta["COMPRESS"] = "LZW"
        # meta["ZLEVEL"] = 1
        meta["TILED"] = True
        # meta["JPEG_QUALITY"] = 90
        # meta["PHOTOMETRIC"] = "YCBCR"
        meta_raster = meta.copy()
        meta_raster["count"] = counts_raster_out
        meta_gt = meta.copy()
        meta_gt["count"] = counts_gt_out
        print(connections)

        for patch_idx, row in zone_gdf.iterrows():

            # print(f"zone idx: {zone_idx}, patch index: {patch_idx}, row patch: {row}")
            bounds = row.geometry.bounds
            # print(bounds)
            window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], meta["transform"])
            patch_transform = transform(window, meta["transform"])
            # print(window)
            # RASTER 2019
            rvb_2019 = connections["rvb_2019"]["conn"].read(window=window)
            # print(rvb_2019.shape)
            irc_2019 = connections["irc_2019"]["conn"].read(indexes=[1], window=window)
            mnh_2019 = connections["mnh_2019"]["conn"].read(indexes=[1], window=window)
            rvb_hm_2019 = connections["rvb_hm_2019"]["conn"].read(window=window)
            raster_2019 = np.vstack([rvb_2019, irc_2019, mnh_2019, rvb_hm_2019])
            raster_2019_path = os.path.join(output_path_zone, *["2019", "raster"])

            if os.path.isdir(raster_2019_path) is False:
                Path(raster_2019_path).mkdir(parents=True, exist_ok=False)

            x = "{:.4f}".format(bounds[0]).replace(".", "-")
            y = "{:.4f}".format(bounds[3]).replace(".", "-")
            filename = f"zone_{zone_idx}_{x}_{y}.tiff"
            print(f"file name {filename}")
            raster_2019_filename = os.path.join(raster_2019_path, filename)
            meta_raster["transform"] = patch_transform
            meta_raster["width"] = raster_2019.shape[1]
            meta_raster["height"] = raster_2019.shape[2]
            # print(raster_2019_filename)
            # print(meta_raster)
            with rio.open(raster_2019_filename, "w+", **meta_raster) as dst:
                dst.write(raster_2019)

            # GT 219
            gt_naf_2019 = connections["gt_naf_2019"]["conn"].read(window=window)
            gt_naf_2019[gt_naf_2019 > 14] = 0
            assert gt_naf_2019[gt_naf_2019 > 14].sum() == 0
            gt_naf_2019_path = os.path.join(output_path_zone, *["2019", "naf"])
            gt_naf_filename = os.path.join(gt_naf_2019_path, filename)
            meta_gt["transform"] = patch_transform
            meta_gt["width"] = gt_naf_2019.shape[1]
            meta_gt["height"] = gt_naf_2019.shape[2]
            gt_urbain_2019 = connections["gt_urbain_2019"]["conn"].read(window=window)
            gt_urbain_2019[gt_urbain_2019 > 8] = 0
            assert gt_urbain_2019[gt_urbain_2019 > 8].sum() == 0
            gt_urbain_2019_path = os.path.join(output_path_zone, *["2019", "urbain"])
            gt_urbain_filename = os.path.join(gt_urbain_2019_path, filename)

            if os.path.isdir(gt_naf_2019_path) is False:
                Path(gt_naf_2019_path).mkdir(parents=True, exist_ok=False)

            if os.path.isdir(gt_urbain_2019_path) is False:
                Path(gt_urbain_2019_path).mkdir(parents=True, exist_ok=False)

            with rio.open(gt_naf_filename, "w+", **meta_gt) as dst:
                dst.write(gt_naf_2019)

            with rio.open(gt_urbain_filename, "w+", **meta_gt) as dst:
                dst.write(gt_urbain_2019)

            # write paths on db
            relative_raster_2019_path = os.path.join(relative_output_path_zone, *["2019", "raster"])
            db.loc[patch_idx, "raster_2019_path"] = os.path.join(relative_raster_2019_path, filename)
            relative_naf_2019_path = os.path.join(relative_output_path_zone, *["2019", "naf"])
            db.loc[patch_idx, "naf_2019_path"] = os.path.join(relative_naf_2019_path, filename)
            relative_urbain_2019_path = os.path.join(relative_output_path_zone, *["2019", "urbain"])
            db.loc[patch_idx, "urbain_2019_path"] = os.path.join(relative_urbain_2019_path, filename)

            ### MAKE 2016 chip where 2016 GT is available
            if updated:

                # RASTER 2016
                rvb_2016 = connections["rvb_2016"]["conn"].read(window=window)
                # print(rvb_2016.shape)
                irc_2016 = connections["irc_2016"]["conn"].read(indexes=[1], window=window)
                mnh_2016 = connections["mnh_2016"]["conn"].read(indexes=[1], window=window)
                rvb_hm_2016 = connections["rvb_hm_2016"]["conn"].read(window=window)
                raster_2016 = np.vstack([rvb_2016, irc_2016, mnh_2016, rvb_hm_2016])
                raster_2016_path = os.path.join(output_path_zone, *["2016", "raster"])

                if os.path.isdir(raster_2016_path) is False:
                    Path(raster_2016_path).mkdir(parents=True, exist_ok=False)

                # filename = f"zone_{zone_idx}_{str(int(bounds[0]))[0:4]}_{str(int(bounds[1]))[0:4]}.tiff"
                raster_2016_filename = os.path.join(raster_2016_path, filename)
                meta_raster["transform"] = patch_transform
                meta_raster["width"] = raster_2016.shape[1]
                meta_raster["height"] = raster_2016.shape[2]

                with rio.open(raster_2016_filename, "w+", **meta_raster) as dst:
                    dst.write(raster_2016)

                # GT 2016d
                gt_naf_2016 = connections["gt_naf_2016"]["conn"].read(window=window)
                gt_naf_2016[gt_naf_2016 > 14] = 0
                assert gt_naf_2016[gt_naf_2016 > 14].sum() == 0
                gt_naf_2016_path = os.path.join(output_path_zone, *["2016", "naf"])
                gt_naf_filename = os.path.join(gt_naf_2016_path, filename)
                meta_gt["transform"] = patch_transform
                meta_gt["width"] = gt_naf_2016.shape[1]
                meta_gt["height"] = gt_naf_2016.shape[2]
                gt_urbain_2016 = connections["gt_urbain_2016"]["conn"].read(window=window)
                gt_urbain_2016[gt_urbain_2016 > 14] = 0
                assert gt_urbain_2016[gt_urbain_2016 > 8].sum() == 0
                gt_urbain_2016_path = os.path.join(output_path_zone, *["2016", "urbain"])
                gt_urbain_filename = os.path.join(gt_urbain_2016_path, filename)

                if os.path.isdir(gt_naf_2016_path) is False:
                    Path(gt_naf_2016_path).mkdir(parents=True, exist_ok=False)

                if os.path.isdir(gt_urbain_2016_path) is False:
                    Path(gt_urbain_2016_path).mkdir(parents=True, exist_ok=False)

                with rio.open(gt_naf_filename, "w+", **meta_gt) as dst:
                    dst.write(gt_naf_2016)

                with rio.open(gt_urbain_filename, "w+", **meta_gt) as dst:
                    dst.write(gt_urbain_2016)

                    # write paths on db
                    relative_raster_2016_path = os.path.join(relative_output_path_zone, *["2016", "raster"])
                    db.loc[patch_idx, "raster_2016_path"] = os.path.join(relative_raster_2016_path, filename)
                    relative_naf_2016_path = os.path.join(relative_output_path_zone, *["2016", "naf"])
                    db.loc[patch_idx, "naf_2016_path"] = os.path.join(relative_naf_2016_path, filename)
                    relative_urbain_2016_path = os.path.join(relative_output_path_zone, *["2016", "urbain"])
                    db.loc[patch_idx, "urbain_2016_path"] = os.path.join(relative_urbain_2016_path, filename)

                # CHANGE 2016
                gt_naf_change = connections["gt_naf_change"]["conn"].read(window=window)
                gt_naf_change_path = os.path.join(output_path_zone, *["change", "naf"])
                gt_naf_filename = os.path.join(gt_naf_change_path, filename)
                gt_urbain_change = connections["gt_urbain_change"]["conn"].read(window=window)
                gt_urbain_change_path = os.path.join(output_path_zone, *["change", "urbain"])
                gt_urbain_filename = os.path.join(gt_urbain_change_path, filename)

                if os.path.isdir(gt_naf_change_path) is False:
                    Path(gt_naf_change_path).mkdir(parents=True, exist_ok=False)

                if os.path.isdir(gt_urbain_change_path) is False:
                    Path(gt_urbain_change_path).mkdir(parents=True, exist_ok=False)

                with rio.open(gt_naf_filename, "w+", **meta_gt) as dst:
                    dst.write(gt_naf_change)

                with rio.open(gt_urbain_filename, "w+", **meta_gt) as dst:
                    dst.write(gt_urbain_change)

                relative_naf_change_path = os.path.join(relative_output_path_zone, *["change", "naf"])
                db.loc[patch_idx, "naf_change_path"] = os.path.join(relative_naf_change_path, filename)
                relative_urbain_change_path = os.path.join(relative_output_path_zone, *["change", "urbain"])
                db.loc[patch_idx, "urbain_change_path"] = os.path.join(relative_urbain_change_path, filename)

    db_file = os.path.join(output_path, "dataset-new.geojson")
    db.to_file(db_file, driver="GeoJSON")


def burn_shape(gdf, class_attribute, shape, class_conv_d, meta):

    classes = gdf[class_attribute].unique() # class present in the shape file
    n_class = len(class_conv_d.keys())
    mask = np.zeros((shape[1], shape[2]))

    for k in classes:

        polys = gdf[gdf[class_attribute] == k]
        polys_to_burn = [(row.geometry, class_conv_d[k]) for idx, row in polys.iterrows()]
        print(polys_to_burn)


def replace_with_dict2(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    return vs[np.searchsorted(ks,ar)]


def replace_with_dict2_generic(ar, dic, assume_all_present=True):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    idx = np.searchsorted(ks,ar)

    if assume_all_present==0:
        idx[idx==len(vs)] = 0
        mask = ks[idx] == ar
        return np.where(mask, vs[idx], ar)
    else:
        return vs[idx]


def rasterize_and_build_change_mask(root_path, db_file,  nomenclature_file):

    db = gpd.read_file(db_file)
    db = db.sort_values(by='2016_updated', ascending=False)
    # db = db[db["id"] == "u1"]
    print(db)
    nomenclature_df = pandas.read_csv(nomenclature_file)
    print(nomenclature_df)
    print(len(db))
    # db = db[db["2016_updated"] == 1]
    print(len(db))
    print(sorted(db["id"].unique()))
    naf_to_urban = {int(r["naf"]): int(r["urbain"]) for ix, r in nomenclature_df.iterrows()}
    naf_to_urban[255] = 255
    print(naf_to_urban)

    for idx, row in tqdm(db.iterrows(), total=len(db)):


        gt_naf = os.path.join(root_path, *[f"zone_{row['id']}", f"zone_{row['id']}_annotation_naf.tif"])
        rvb = os.path.join(root_path, *[f"zone_{row['id']}", f"zone_{row['id']}_rvb.tif"])

        with rio.open(rvb) as src:

            src_transform = src.meta["transform"]
            src_shape = int(src.meta["height"]), int(src.meta["width"])
            profile = src.profile.copy()
            profile["count"] = 1
            print(profile)

        with rio.open(gt_naf) as src:

            msk_naf = src.read()

        polygons = []
        # msk_urban = np.zeros(msk_naf.shape)

        # convert gt naf to urban
        print(row["id"])
        msk_urban = replace_with_dict2_generic(msk_naf, naf_to_urban).astype("uint8")
        print(np.unique(msk_urban))
        # exit(0)
        # msk urban should not contains 0 value
        assert msk_urban[msk_urban <= 0].astype("uint8").sum() == 0

        # write urban on disk for 2019
        urban_filename = os.path.join(PATH_TO_2019, *[f"zone_{row['id']}", f"zone_{row['id']}_annotation_urbain.tif"])

        with rio.open(urban_filename, "w+", **profile) as dst:

            dst.write(msk_urban)

        if row["2016_updated"] == 1:

            gt = os.path.join(root_path, *[f"zone_{row['id']}", f"zone_{row['id']}_saisie.shp"])
            gt_gdf = gpd.read_file(gt)
            # print(len(gt_gdf))
            gt_gdf.dropna(axis=0, subset=['class'], inplace=True)
            print(len(gt_gdf))
            print(f" row: {row['id']} updated")

            # burn updated polygons
            for idx1, row1 in gt_gdf.iterrows():

                if row1["class"] != "inconnu":

                    # print(row1["class"])
                    value = nomenclature_df.loc[nomenclature_df["legende"] == row1["class"], "naf"]
                    value = int(value.iloc[0])
                    polygons.append((row1.geometry, value))
                    # print(len(polygons))

            msk_naf_updated = features.rasterize(polygons,
                                     out_shape=src_shape,
                                     transform=src_transform,
                                     fill=0,
                                     all_touched=True)

            msk_naf_updated = np.expand_dims(msk_naf_updated, axis=0)
            print(f"source shape {src_shape}")
            print(f"msk naf updated shape {msk_naf_updated.shape}")
            print(f"msk naf shape {msk_naf.shape}")
            # convert gt naf to urban

            # exit(0)
            msk_naf_change = (msk_naf_updated > 0).astype("uint8")
            msk_naf_updated = np.where(msk_naf_change > 0, msk_naf_updated, msk_naf)
            msk_urban_updated = replace_with_dict2_generic(msk_naf_updated, naf_to_urban,
                                                           assume_all_present=False).astype("uint8")
            msk_urban_change = (msk_urban_updated != msk_urban).astype("uint8")
            # write urban on disk for 2016
            urban_filename_updated = os.path.join(root_path, *[f"zone_{row['id']}", f"zone_{row['id']}_annotation_urbain_updated.tif"])
            naf_filename_updated = os.path.join(root_path, *[f"zone_{row['id']}", f"zone_{row['id']}_annotation_naf_updated.tif"])
            urban_filename_change = os.path.join(root_path, *[f"zone_{row['id']}", f"zone_{row['id']}_annotation_urbain_change.tif"])
            naf_filename_change = os.path.join(root_path, *[f"zone_{row['id']}", f"zone_{row['id']}_annotation_naf_change.tif"])

            with rio.open(urban_filename_updated, "w+", **profile) as dst:

                dst.write(msk_urban_updated)

            with rio.open(naf_filename_updated, "w+", **profile) as dst:

                dst.write(msk_naf_updated)

            with rio.open(urban_filename_change, "w+", **profile) as dst:

                dst.write(msk_urban_change)

            with rio.open(naf_filename_change, "w+", **profile) as dst:

                dst.write(msk_naf_change)


def copy_update_from_2016():

    update_vector_layers_path = os.path.join(NAS, "DETECT_CHANGE/gers/C_RETOUR_2")
    shape_file_pattern = "*/projet_qgs_ini/FINAL_DATA/OCSGE-DL/*Final.*"
    shape_file_pattern_2 = "*/FINAL_DATA/OCSGE-DL/*Final.*"
    zone_gdf = gpd.read_file("/media/HP-2007S005-data/gers/zones_gdf.geojson")
    zone_gdf["2016_updated"] = 0
    updated_vector_layers = glob.glob(os.path.join(update_vector_layers_path, shape_file_pattern))
    updated_vector_layers_2 = glob.glob(os.path.join(update_vector_layers_path, shape_file_pattern_2))
    print(f" updated vetcors pattern 1 {updated_vector_layers}")
    print(f" updated vetcors pattern 2 {updated_vector_layers_2}")
    # updated_vector_layers = updated_vector_layers
    # zones_updated = {i.split("/")[-5]: i for i in updated_vector_layers}
    zones_updated = {}

    for i in updated_vector_layers:

        zone = i.split("/")[-5]
        ext = i.split(".")[-1]

        if zone in zones_updated.keys():

            zones_updated[zone][ext] = i

        else:

            zones_updated[zone] = {ext: i}

    zones_updated_2 = {}

    for i in updated_vector_layers_2:

        zone = str(i.split("/")[-4]).lower()
        ext = i.split(".")[-1]

        if zone in zones_updated_2.keys():

            zones_updated_2[zone][ext] = i

        else:

            zones_updated_2[zone] = {ext: i}

    zones_updated = dict(zones_updated, **zones_updated_2)
    print(zones_updated)
    print(sorted(zones_updated.keys()))
    print(len(zones_updated))
    gt_2016 = glob.glob(os.path.join(PATH_TO_2016, f"*/{GT_FILE}"))
    print(f"ground truth 2016 {gt_2016}")
    zones_2016 = {i.split("/")[-2].split("_")[1]: i.split(".")[0] for i in gt_2016}
    print(f" zones 2016 {zones_2016}")
    print(f"length of zones 2016 {len(zones_2016)}")

    for zone, data in tqdm(zones_updated.items()):

        path_dst = zones_2016[zone]
        print(path_dst)
        zone_gdf.loc[zone_gdf["id"] == zone, "2016_updated"] = 1
        print(zone_gdf[zone_gdf["id"] == zone])

        for ext, f in data.items():

            file_dst = path_dst + "." + f.split(".")[-1]
            # print(file_dst)
            copy2(f, file_dst)

    fold_cols = ["fold_1_train", "fold_1_val", "fold_2_train", "fold_2_val", "fold_3_train", "fold_3_val"]
    fold_1 = ["n11", "u2", "v12"]
    fold_2 = ["n19", "u5", "v1"]
    fold_3 = ["n17", "u1", "v10"]
    zone_gdf[fold_cols] = [0, 0, 0, 0, 0, 0]
    zone_gdf["fold_1_train"] = zone_gdf.apply(lambda x: 0 if x["id"] in fold_1 else 1, axis=1)
    zone_gdf["fold_1_val"] = zone_gdf.apply(lambda x: 1 if x["id"] in fold_1 else 0, axis=1)
    zone_gdf["fold_2_train"] = zone_gdf.apply(lambda x: 0 if x["id"] in fold_2 else 1, axis=1)
    zone_gdf["fold_2_val"] = zone_gdf.apply(lambda x: 1 if x["id"] in fold_2 else 0, axis=1)
    zone_gdf["fold_3_train"] = zone_gdf.apply(lambda x: 0 if x["id"] in fold_3 else 1, axis=1)
    zone_gdf["fold_3_val"] = zone_gdf.apply(lambda x: 1 if x["id"] in fold_3 else 0, axis=1)
    zone_gdf["test"] = zone_gdf["id"].apply(lambda x: x in tests_zones) # set test zones
    zone_type = {"n": "natuelle", "v": "végétale", "u": "urbain"}
    zone_gdf["zone_type"] = zone_gdf["id"].apply(lambda x: zone_type[x[0]])
    print(zone_gdf[zone_gdf["2016_updated"] == 1])
    print(len(zone_gdf[zone_gdf["2016_updated"] == 1]))
    zone_gdf.to_file("/media/HP-2007S005-data/gers/zones_gdf-new.geojson", driver="GeoJSON")


def compute_stats_supervised_dataset(dataset_file, nomenclature_file, root):

    gdf_dataset = gpd.read_file(dataset_file)
    df_nomenclature = pd.read_csv(nomenclature_file)
    channel_raster = ["R", "V", "B", "IR", "MNH", "R-HM", "G-HM", "B-HM"]
    class_naf = {int(row["naf"]): row["legende_naf"] for i, row in df_nomenclature.iterrows()}
    class_naf[0] = "no-label"
    class_urbain = {int(row["urbain"]): row["legende_urbain"] for i, row in df_nomenclature.iterrows()}
    class_urbain[0] = "no-label"
    gt_of_interest = ["naf_2019_path", "urbain_2019_path"]
    change_of_interest = [{"path": "naf_change_path", "col": "naf_change_per"},
                          {"path": "urbain_change_path", "col": "urbain_change_per"}]
    print(f" class urbain: {class_urbain}")
    print(f" class naf: {class_naf}")
    histo_raster_2019 = {"path": "raster_2019_path", "stats": {channel: {"histo": np.zeros((256)), "count": 0, "sum_L1": 0, "sum_L2": 0} for channel in channel_raster}}
    histo_raster_2016 = {"path": "raster_2016_path", "stats": {channel: {"histo": np.zeros((256)), "count": 0, "sum_L1": 0, "sum_L2": 0} for channel in channel_raster}}
    counter_2019 = 0
    counter_2016 = 0
    # bins = None

    for idx, row in tqdm(gdf_dataset.iterrows(), total=len(gdf_dataset)):

        counter_2019 += 1
        r_2019_path = os.path.join(root, row[histo_raster_2019["path"]])

        with rio.open(r_2019_path) as src:

            bands = src.read()
            img = reshape_as_image(bands)

            for i, (channel, value) in enumerate(histo_raster_2019["stats"].items()):

                # histo, count, sum_l1, sum_l2 = value["histo"], value["count"], value["sum_L1"], value["sum_L2"]
                # print(i)
                n = cv2.calcHist([img], [i], None, [256], [0, 256])

                data = img[:, :, i]
                # n, c_bins = np.histogram(data)
                # print(f"tmp histo {tmp_histo} channel {channel}")
                # print(f"tmp histo norm {(tmp_histo / 255).max()}")
                print(f'count: {histo_raster_2019["stats"][channel]["count"]}')
                histo_raster_2019["stats"][channel]["histo"] = np.add(histo_raster_2019["stats"][channel]["histo"], n)
                histo_raster_2019["stats"][channel]["count"] += int(np.prod(data.shape))
                histo_raster_2019["stats"][channel]["sum_L1"] += int(data.sum())
                histo_raster_2019["stats"][channel]["sum_L2"] += int((data ** 2).sum())
                # print(data.shape)
                # print(np.prod(data.shape))
                print(f"simple sum {data.sum()}")
                print(f"square sum {(data ** 2).sum()}")


                """
                if histo is not None:
                    print(histo.sum())
                    print(histo.shape)
                """

        for gt in gt_of_interest:

            gt_path = os.path.join(root, row[gt])
            type_change = "naf" if "naf" in gt else "urbain"
            classes = class_naf if "naf" in gt else class_urbain

            with rio.open(gt_path) as src:

                bands = src.read()
                img = reshape_as_image(bands)
                tot = img.shape[0] * img.shape[1] * img.shape[2]
                tmp_histo = cv2.calcHist([img], [0], None, [len(classes)], [0, len(classes)])
                # print(tmp_histo.shape)
                # exit(0)
                for i, value in enumerate(tmp_histo[:, 0]):
                    if type_change == "naf" and i == 7:
                        pass
                    else:
                        gdf_dataset.loc[idx, f"{classes[i]}_{type_change}_freq" ] = value / tot
                        # print(f"{classes[int(i)]}_{type_change}_freq : {value / tot}")

        is_updated = row["2016_updated"]
        if is_updated:

            counter_2016 += 1
            r_2016_path = os.path.join(root, row[histo_raster_2016["path"]])

            with rio.open(r_2016_path) as src:

                bands = src.read()
                img = reshape_as_image(bands)

                for i, (channel, value) in enumerate(histo_raster_2016["stats"].items()):
                    # histo, count, sum_l1, sum_l2 = value["histo"], value["count"], value["sum_L1"], value["sum_L2"]
                    # print(i)
                    n = cv2.calcHist([img], [i], None, [256], [0, 256])
                    data = img[i]
                    # n, c_bins = np.histogram(data, range(257))
                    # print(f"tmp histo {tmp_histo} channel {channel}")
                    # print(f"tmp histo norm {(tmp_histo / 255).max()}")
                    histo_raster_2016["stats"][channel]["histo"] = np.add(histo_raster_2016["stats"][channel]["histo"], n)
                    histo_raster_2016["stats"][channel]["count"] += int(np.prod(data.shape))
                    histo_raster_2016["stats"][channel]["sum_L1"] += int(data.sum())
                    histo_raster_2016["stats"][channel]["sum_L2"] += int((data**2).sum())

            for change in change_of_interest:

                change_path = os.path.join(root, row[change["path"]])
                with rio.open(change_path) as src:
                    bands = src.read()
                    tot = bands.shape[0] * bands.shape[1] * bands.shape[2]
                    q_change = bands[bands > 0].sum()
                    freq_change = q_change / tot
                    gdf_dataset.loc[idx, change["col"]] = freq_change
                    # print(freq_change)

    gdf_dataset.to_file(os.path.join(root, "supervised_dataset_with_stats.geojson"), driver="GeoJSON")
    print(counter_2019)
    print(counter_2016)

    for i, (channel, value) in enumerate(histo_raster_2019["stats"].items()):

        histo, count, sum_l1, sum_l2 = value["histo"], value["count"], value["sum_L1"], value["sum_L2"]
        mean = sum_l1 / count
        var = (sum_l2 / count) - (mean ** 2)
        std = np.sqrt(var)
        histo_raster_2019["stats"][channel] = dict()
        histo_raster_2019["stats"][channel]["mean"] = mean
        histo_raster_2019["stats"][channel]["var"] = var
        histo_raster_2019["stats"][channel]["std"] = std
        histo_raster_2019["stats"][channel]["sum_L1"] = sum_l1
        histo_raster_2019["stats"][channel]["sum_L2"] = sum_l2
        histo_raster_2019["stats"][channel]["count"] = count
        # histo_raster_2019["stats"][channel]["histo"] = histo.tolist()
        print(histo_raster_2019["stats"][channel]["mean"])
        print(histo_raster_2019["stats"][channel]["std"])

    for i, (channel, value) in enumerate(histo_raster_2016["stats"].items()):
        histo, count, sum_l1, sum_l2 = value["histo"], value["count"], value["sum_L1"], value["sum_L2"]
        mean = sum_l1 / count
        var = (sum_l2 / count) - (mean ** 2)
        std = np.sqrt(var)
        histo_raster_2016["stats"][channel]["mean"] = mean
        histo_raster_2016["stats"][channel]["var"] = var
        histo_raster_2016["stats"][channel]["std"] = std
        histo_raster_2016["stats"][channel]["histo"] = histo.tolist()
        print(histo_raster_2016["stats"][channel]["mean"])
        print(histo_raster_2016["stats"][channel]["std"])

    with open(os.path.join(root, "raster_2016_stats.json"), 'w') as fp:
        json.dump(histo_raster_2016, fp)
    with open(os.path.join(root, "raster_2019_stats.json"), 'w') as fp:
        json.dump(histo_raster_2019, fp)


def get_worksite_by_date(path_date_1, path_date_2, output_path, crs="epsg:2154"):

    update_vector_layers_path = path_date_1
    shape_file_pattern = "*/projet_qgs_ini/FINAL_DATA/OCSGE-DL/*Final.shp"
    shape_file_pattern_2 = "*/FINAL_DATA/OCSGE-DL/*Final.shp"
    zone_gdf = gpd.read_file("/media/HP-2007S005-data/gers/zones_gdf.geojson")
    zone_gdf["2016_updated"] = 0
    updated_vector_layers = glob.glob(os.path.join(update_vector_layers_path, shape_file_pattern))
    updated_vector_layers_2 = glob.glob(os.path.join(update_vector_layers_path, shape_file_pattern_2))
    updated_vector_T0 = updated_vector_layers + updated_vector_layers_2
    updated_vector_T0 = {p.split("/")[-1].split("_")[2]: p for p in updated_vector_T0}
    # print(updated_vector_layers)
    # print(updated_vector_layers_2)
    print(len(updated_vector_T0))
    # print(updated_vector_T0)
    # print(sorted(updated_vector_T0.keys()))
    shape_file_pattern = "*/01_LIVRAISONS/*.shp"
    shape_file_pattern_2 = "*/01_LIVRAISON/*.shp"

    updated_vector_T1 = glob.glob(os.path.join(path_date_2, shape_file_pattern)) + glob.glob(os.path.join(path_date_2, shape_file_pattern_2))
    print(len(updated_vector_T1))
    # updated_vector_T1 = {p.split("/")[-1].split("_")[2].lower(): p for p in updated_vector_T1}
    d = dict()
    for p in updated_vector_T1:
        sp = p.split("/")[-1].split("_")
        key = sp[2].lower()
        if key in d.keys():
            if len(sp) == 5:
                print(f"updated key {key} with sp {sp}")
                d[key] = p
        else:
            d[key] = p


    updated_vector_T1 = d
    inter_keys = sorted(list(set(updated_vector_T0) & set(updated_vector_T1)))
    updated_vector_T0 = {k: v for k, v in updated_vector_T0.items() if k in inter_keys}
    updated_vector_T1 = {k: v for k, v in updated_vector_T1.items() if k in inter_keys}
    # print(updated_vector_T1)
    # print(sorted(updated_vector_T1.keys()))
    print(inter_keys)
    print(len(inter_keys))
    print(len(updated_vector_T1))
    print(len(updated_vector_T0))

    annotations = {k: {"2016": v, "2019": updated_vector_T1[k]} for k, v in updated_vector_T0.items()}
    with open(os.path.join(output_path, "annotations.json"), "w") as f:
        json.dump(annotations, f)
    diff = list()
    for k, v in tqdm(updated_vector_T0.items()):
        gdf = gpd.read_file(v)
        print(f" length of gdf before drop: {len(gdf)}")
        gdf.dropna(axis=0, subset=['class'], inplace=True)
        gdf["zone"] = k
        print(f" length of gdf after drop: {len(gdf)}")
        diff.append(gdf)
    diff_df = pd.concat(diff)
    diff_gdf = gpd.GeoDataFrame(diff_df, geometry="geometry", crs = crs)
    diff_gdf.to_file(os.path.join(output_path, "diff_annotation_gers.shp"))


def merge_annotations_2019(annotations_2019, annotations_2016, output_path):
    annotations_2016 = gpd.read_file(annotations_2016)
    annotations_2019 = gpd.read_file(annotations_2019)
    print(f"length of annotation 2016 {len(annotations_2016)}")
    print(f"length of annotation 2019 {len(annotations_2019)}")
    joined_gdf = gpd.sjoin(annotations_2016, annotations_2019, op="intersects")
    print(f"length of joined gdf {len(joined_gdf)}")
    joined_gdf.to_file(output_path)
    l = list()
    """
    with open(json_file) as fp:
        annotations = dict(json.load(fp))
    for k, v in tqdm(annotations.items()):
        T1 = gpd.read_file(v['2019'])
        l.append(T1)
    gdf = gpd.GeoDataFrame(pd.concat(l), crs=annotation_2016.crs, geometry='geometry')
    gdf.to_file('/home/ign.fr/skhelifi/data/gers/annotations_gers/annotation_2019_on_diff.shp')
    """

def convert_field(field):
    if field in nomenclature_v1.keys():
        return nomenclature_v1[field]
    else: return field


def reduce_change(input_file, output_file):
    input_gdf = gpd.read_file(input_file)
    crs = input_gdf.crs
    input_gdf = input_gdf[~input_gdf['class'].isin(['inconnu'])]
    print(input_gdf['class'].unique())
    print(input_gdf['libelle'].unique())
    print(set(input_gdf['class'].unique()).difference(set(input_gdf['libelle'].unique())))
    input_gdf['class'] = input_gdf['class'].apply(convert_field)
    input_gdf['libelle'] = input_gdf['libelle'].apply(convert_field)
    print(input_gdf['class'].unique())
    print(input_gdf['libelle'].unique())
    input_gdf['libelle'] = input_gdf['libelle'].apply(lambda x: "feuillus" if x == "feuillu" else x)
    print(set(input_gdf['class'].unique()).difference(set(input_gdf['libelle'].unique())))
    print(len(input_gdf))

    input_gdf = input_gdf[input_gdf['class'] != input_gdf['libelle']]
    print(len(input_gdf))
    print(input_gdf['class'].unique())
    print(input_gdf['libelle'].unique())
    input_gdf = gpd.GeoDataFrame(input_gdf, crs=crs, geometry="geometry")
    input_gdf.to_file(output_file)


def check_nomenclature(json_file):
    labels = set()
    with open(json_file) as fp:
        annotations = dict(json.load(fp))
    for k, v in tqdm(annotations.items()):
        T0 = gpd.read_file(v["2016"])
        T0.dropna(axis=0, subset=['class'], inplace=True)
        T1 = gpd.read_file(v["2019"])
        labels = labels.union(T0["class"].unique(), T1["libelle"].unique())
        if len(T0) != len(T1):
            print(f"length different for zone {k}, TO {len(T0)}, T1: {len(T1)}")
        idx_T0 = T0["label"].unique()
        idx_T1 = T1["id"].unique()
        print(f"difference between T0 and T1 {len(set(idx_T0) - set(idx_T1))}")
    print(labels)


def filter_change_map(change_map, output_path):

    gdf = gpd.read_file(change_map)
    crs =gdf.crs
    gdf = gpd.GeoDataFrame(gdf[["id", "zone_type", "geometry"]], crs=crs, geometry="geometry")
    gdf.to_file(os.path.join(output_path, "diff_annotations_merged.shp"))


def merge_zone_and_supervised_dataset(zone_path: str, dataset_path: str) -> None:
    """
    :param zone_path:
    :param dataset_path:
    :return:
    """
    zone_gdf = gpd.read_file(zone_path)
    supervised_gdf = gpd.read_file(dataset_path)
    print(supervised_gdf.columns)
    cols_to_drop = [ 'rvb_2019', 'rvb_fdm_2019', 'rvb_hm_2019',
       'rvb_mm_2019', 'rvb_2016', 'rvb_fdm_2016', 'rvb_hm_2016', 'rvb_mm_2016',
       'irc_2019', 'irc_fdm_2019', 'irc_hm_2019', 'irc_mm_2019', 'irc_2016',
       'irc_fdm_2016', 'irc_hm_2016', 'irc_mm_2016', 'mnh_2016', 'mnh_2019',
       'gt_2019', 'gt_2016']
    supervised_gdf.drop(columns=cols_to_drop, inplace=True)
    print(zone_gdf)
    print(supervised_gdf)
    print(zone_gdf.columns)
    cols = ['fold_1_train', 'fold_1_val', 'fold_2_train', 'fold_2_val', 'fold_3_train', 'fold_3_val']
    supervised_gdf[cols] = [0, 0, 0, 0, 0, 0]
    train_gdf = supervised_gdf[supervised_gdf["test"] == 0]

    for idx, row in tqdm(train_gdf.iterrows(), total=len(train_gdf)):

        zone_id = row["id"]
        supervised_gdf.loc[idx, cols] = zone_gdf.loc[zone_gdf["id"] == zone_id, cols].iloc[0]

    output_file = dataset_path.split(".")[0] + "-new.geojson"
    supervised_gdf.to_file(output_file, driver="GeoJSON")


def set_weights_for_sampler(dataset_file, output_file):

    dataset = gpd.read_file(dataset_file)
    urban_cols = ['no-label_urbain_freq',
                  'batiment_urbain_freq', 'ligneux_urbain_freq', 'herbacee_urbain_freq',
                  'bitume_urbain_freq', 'mineraux_urbain_freq', 'sol_nus_urbain_freq',
                  'eau_urbain_freq', 'piscine_urbain_freq']
    naf_cols = ['no-label_naf_freq', 'batiment_naf_freq', 'zone_permeable_naf_freq',
                'zone_impermeable_naf_freq', 'piscine_naf_freq', 'sol_nu_naf_freq',
                'surface_eau_naf_freq', 'coniferes_naf_freq', 'coupe_naf_freq',
                'feuillus_naf_freq', 'broussaille_naf_freq', 'vigne_naf_freq',
                'culture_naf_freq', 'terre_labouree_naf_freq']
    freq_urban_cols = ['ligneux_urbain_freq', 'herbacee_urbain_freq']
    freq_naf_cols = ['culture_naf_freq', 'feuillus_naf_freq']
    non_freq_urban_cols = [i for i in urban_cols if i not in freq_urban_cols]
    non_freq_naf_cols = [i for i in naf_cols if i not in freq_naf_cols]
    dataset["weight_urbain"] = dataset.apply(lambda x: ((x[non_freq_urban_cols].sum() * 10) ** 2) + 1, axis=1)
    dataset["weight_naf"] = dataset.apply(lambda x: ((x[non_freq_naf_cols].sum() * 10) ** 2) + 1, axis=1)
    dataset.to_file(os.path.join(output_file), driver="GeoJSON")


def polygonize_mask(db_file: str,
                    path_to_data: str,
                    raster_cols:  Dict,
                    driver: str = "GeoJSON",
                    output_dir="/home/ign.fr/skhelifi/data/gers") -> None:
    print(driver)

    gdf = gpd.read_file(os.path.join(path_to_data, db_file))
    gdf = gdf[gdf["2016_updated"] == 1]
    for col_name, col in raster_cols.items():
        d = list()
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
            with rio.open(os.path.join(path_to_data, row[col])) as src:
                mask = src.read().astype("uint8")
                transform = src.meta["transform"]
                for polygon, value in rio.features.shapes(mask, mask=mask, transform=transform):
                    sp = row[col].split("/")
                    uuid_name = sp[3].split(".")[0]
                    uuid = f"{sp[0]}-{sp[2]}-{uuid_name}"
                    d_row = {"geometry": shape(polygon), "cl_value": value, "uid": uuid}
                    d.append(d_row)

        output_file = f"{col_name}.geojson" if driver == "GeoJSON" else f"{col_name}.shp"
        output_gdf = gpd.GeoDataFrame(d, crs=gdf.crs, geometry="geometry")
        output_gdf.to_file(os.path.join(output_dir, output_file), driver=driver)


def filter_mask(f: str, f_out:str,min_area_size:int=50) -> None:

    gdf = gpd.read_file(f)
    gdf_out: gpd.GeoDataFrame = gdf[gdf['geometry'].apply(lambda x: x.area >= min_area_size)]
    print(len(gdf))
    print(len(gdf_out))
    gdf_out.to_file(f_out)


def clean_differentiel(diff_annotations: str, annotations_2019: str, diff_ocsge: str, extent: str, output_path: str) -> gpd.geodataframe:

    diff_annotations = gpd.read_file(diff_annotations)
    diff_ocsge = gpd.read_file(diff_ocsge)
    extent = gpd.read_file(extent)
    """
    print(len(diff_annotations))
    diff_annotations = gpd.sjoin(diff_annotations, extent, op="within")
    print(len(diff_annotations))
    diff_annotations.to_file("/home/ign.fr/skhelifi/data/gers/annotations_gers/diff_annotaions_cleaned.shp")
    """
    field = "class"
    non_pertinent_change = {
        "": ""
    }
    print()


if __name__ == "__main__":
    """
    copy_update_from_2016() # copy shapefiles from 2016 Terria store to database
    rasterize_and_build_change_mask(PATH_TO_2016,
                                    os.path.join(ROOT_OUT, "gers/zones_gdf.geojson"),
                                    os.path.join(ROOT_OUT, "gers/nomenclature.csv"))
    # generate_dataset(gdf_file=os.path.join(GERS_DATA_PATH, *["32_256X256_supervised_sample_global", "supervised_sample_256cm_gers_area.geojson"]), zone_file=os.path.join(GERS_DATA_PATH, "zones_gdf.geojson"))
    make_chips(os.path.join(ROOT_OUT, "gers/supervised_dataset_db.geojson"),
               os.path.join(ROOT_OUT, "gers/supervised_dataset"))

    compute_stats_supervised_dataset(os.path.join(ROOT_OUT, "gers/supervised_dataset/dataset.geojson"),
                                     os.path.join(ROOT_OUT, "gers/nomenclature.csv"),
                                     os.path.join(ROOT_OUT, "gers/supervised_dataset"))

    set_weights_for_sampler(os.path.join(ROOT_OUT, "gers/supervised_dataset/supervised_dataset_with_stats.geojson"),
                            os.path.join(ROOT_OUT, "gers/supervised_dataset/supervised_dataset_with_stats_and_weights.geojson"))
    """
    """
    polygonize_mask("supervised_dataset_with_stats_and_weights.geojson",
                    path_to_data=os.path.join(ROOT_OUT, "gers/supervised_dataset"),
                    raster_cols={"naf_change": "naf_change_path",
                                 "urbain_change": "urbain_change_path"})
    """
    """
    polygonize_mask("supervised_dataset_with_stats_and_weights.geojson",
                    path_to_data=os.path.join(ROOT_OUT, "gers/supervised_dataset"),
                    raster_cols={"naf_2019": "naf_2019_path",
                                 "naf_2016": "naf_2016_path",
                                 "urbain_2019": "urbain_2019_path",
                                 "urbain_2016": "urbain_2016_path"},
                    driver='ESRI Shapefile')
    """
    """
    filter_mask("/home/ign.fr/skhelifi/data/gers/diff_prod_gers_2016-2019/DIFF_2016_2019.shp",
                 "/home/ign.fr/skhelifi/data/gers/diff_prod_gers_2016-2019/DIFF_2016_2019_filtered_area.shp",
                 50)
    """
    path_date_1 = "/media/HP-2007S005-media/NAS/DETECT_CHANGE/gers/chantiers/chantier_annotation_2016"
    path_date_2 = "/media/HP-2007S005-media/NAS/DETECT_CHANGE/gers/chantiers/chantier_annotation_2019"
    output_path = "/home/ign.fr/skhelifi/data/gers/annotations_gers"
    # get_worksite_by_date(path_date_1, path_date_2, output_path)
    # check_nomenclature("/media/HP-2007S005-media/NAS/DETECT_CHANGE/gers/annotations.json")
    #filter_change_map(change_map="/media/HP-2007S005-data/gers/supervised_dataset/naf_change.geojson",
    #                  output_path="/home/ign.fr/skhelifi/data/gers/annotations_gers")
    diff_annotations = "/media/HP-2007S005-media/NAS/DETECT_CHANGE/gers/diff_annotation_gers.shp"
    diff_ocsge = "/home/ign.fr/skhelifi/data/gers/diff_prod_gers_2016-2019/DIFF_2016_2019_filtered_area.shp"
    output_path = "/home/ign.fr/skhelifi/data/gers/annotations_gers/diff_annotations_cleaned.shp"
    #merge_annotations_2019("/home/ign.fr/skhelifi/data/gers/annotations_gers/annotation_2019_on_diff.shp", "/home/ign.fr/skhelifi/data/gers/annotations_gers/diff_annotaions_cleaned.shp",
    #                      "/home/ign.fr/skhelifi/data/gers/annotations_gers/naf_diff_raw.shp")
    # clean_differentiel(diff_annotations, diff_ocsge, extent="/home/ign.fr/skhelifi/data/departements/deps/dep_32.shp",output_path=output_path)
    reduce_change("/home/ign.fr/skhelifi/data/gers/annotations_gers/naf_diff_raw.shp",
                  "/home/ign.fr/skhelifi/data/gers/annotations_gers/naf_diff_reduced.shp")
