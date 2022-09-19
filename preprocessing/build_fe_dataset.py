import glob
import os
from typing import Tuple

# from rasterio.windows import from_bounds as window_from_bounds
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features
from rasterio.transform import from_bounds
# from shapely.geometry import shape
from shapely.geometry import box
from tqdm import tqdm

# import random

db_output_file = "dataset.shp"
RESAMPLING = rio.enums.Resampling.nearest
BOUNDLESS = True
remote = True
OUTPUT_PATH = "/media/HP-2007S005-data/dataset_fe" if remote is False else "/var/data/dl/dataset_fe"
CROSS_LABEL_MATRIX = {1: 1,
                      2: 2,
                      3: 3,
                      4: 4,
                      5: 5,
                      6: 6,
                      7: 7,
                      8: 8,
                      9: 0,
                      10: 9,
                      11: 0,
                      12: 0,
                      13: 10,
                      14: 11,
                      15: 12,
                      16: 12,
                      17: 12,
                      18: 13,
                      19: 0}

CROSS_LABEL_MATRI_PROD_NAF_CHANGE = {10: 1,
                                     12: 2,
                                     11: 3,
                                     13: 4,
                                     21: 5,
                                     22: 6,
                                     23: 7,
                                     302: 8,
                                     300: 0,
                                     301: 9,
                                     303: 0,
                                     30: 0,
                                     31: 10,
                                     32: 11,
                                     42: 12,
                                     41: 12,
                                     40: 12,
                                     101: 13,
                                     0: 0}

ROOT = "/media/DATA10T_1/TERRIA/DATASETS_DAI"
ZONE_LIST = ""
ORTHO = os.path.join(ROOT, "ORTHO")
DEM = os.path.join(ROOT, "DEM")
LABELS_VECT = os.path.join(ROOT, "LABELS_VECT")


def run_process_1():
    """
    IN: zone list shapefile, path_to_data

    OUT:
    Step 1/ merge data from zones: RVB, IR, MNS, MNT, VECTOR LABEL

    Step 2/ For each zone in zones:
        Substep 1/ compute transform
        SubStep 2/ get zone change shape file and burn it in mask directory and save
        it and update shapfile dataset
        Substep 3/ get RVB, IRC  and compute MNH from MNS and MNT for T0 and T1 and burn it and save
        it in image directory and update shapfile dataset

    Returns
    -------

    """
    def get_zone_id_date_dep(name: str) -> Tuple[str, str, str]:

        sp = name.split(".")[0].split("/")[-1].split("_")
        zone = "_".join([sp[0], sp[1], sp[2], sp[3], sp[4]])
        date = sp[1]
        dep = sp[0]
        return zone, date, dep

    rvb_list = glob.glob(os.path.join(ORTHO, "*/*/*_RVB.tif"))
    rvb_dict = {get_zone_id_date_dep(path)[0]: path for path in rvb_list}
    ir_list = glob.glob(os.path.join(ORTHO, "*/*/*_IR.tif"))
    ir_dict = {get_zone_id_date_dep(path)[0]: path for path in ir_list}
    mns_list = glob.glob(os.path.join(DEM, "*/*/*_DSM.tif"))
    mns_dict = {get_zone_id_date_dep(path)[0]: path for path in mns_list}
    mnt_list = glob.glob(os.path.join(DEM, "*/*/*_DTM.tif"))
    mnt_dict = {get_zone_id_date_dep(path)[0]: path for path in mnt_list}
    vector_list = glob.glob(os.path.join(LABELS_VECT, "*/*/*.shp"))
    vector_dict = [{"zone_id": get_zone_id_date_dep(path)[0],
                    "vector_path": path,
                    "year": get_zone_id_date_dep(path)[1],
                    "dep": get_zone_id_date_dep(path)[2]} for path in vector_list]

    print(len(rvb_list))
    print(len(ir_list))
    print(len(mns_list))
    print(len(mnt_list))
    print(len(vector_list))
    # print(rvb_dict)
    # print(vector_dict)
    # print(ir_dict)
    crs = None
    #  l = [v.update({"irc_path": ir_dict[v["zone_id"]]}) for v in vector_dict]
    # print(l)
    l_poly = []
    for v in vector_dict:
        # print(ir_dict[v["zone_id"]])
        if v["zone_id"] in mns_dict.keys():
            l_poly.append({"zone_id": v["zone_id"],
                           "vector_path": v["vector_path"],
                           "year": v["year"],
                           "dep": v["dep"],
                           "ir_path": ir_dict[v["zone_id"]],
                           "mns_path": mns_dict[v["zone_id"]],
                           "mnt_path": mnt_dict[v["zone_id"]],
                           "rvb_path": rvb_dict[v["zone_id"]]})

    # print(l)

    # Step 2/ For each zone in zones:
    # l = random.sample(l, 5)
    # print(l)
    target_resolution = 0.2

    meta: dict = {"driver": "GTiff",
                  "crs": crs,
                  "TILED": "YES",
                  "dtype": "uint8"}
    l2 = []

    for v in tqdm(l_poly):

        try:

            # print(v["zone_id"])
            # print(v["rvb_path"])
            # print(v["ir_path"])
            # print(v["mns_path"])
            # print(v["mnt_path"])

            # Substep 1/ compute transform
            gdf = gpd.read_file(v["vector_path"])
            # print("file read")
            # main_area = gdf.area.sum()
            # print(f"main area: {main_area}")
            # gdf["area"] = gdf.apply(lambda x: x["geometry"].area, axis=1)
            # print("area computed")
            # grouped_gdf = gdf.groupby(["code"]).agg({'area': ['sum']})
            # grouped_gdf.columns = grouped_gdf.columns.droplevel(0)
            # grouped_gdf.reset_index(inplace=True)
            # print(f"grouped gdf columns: {grouped_gdf.columns}")
            crs = gdf.crs if crs is None else crs
            bounds = gdf.unary_union.bounds
            x_min, y_min, x_max, y_max = bounds[0], bounds[1], bounds[2], bounds[3]
            poly_box = box(x_min, y_min, x_max, y_max)
            width = int((x_max - x_min) / target_resolution)
            height = int((y_max - y_min) / target_resolution)
            out_shape = (height, width)
            out_transform = from_bounds(west=x_min, south=y_min, east=x_max, north=y_max, width=width, height=height)

            # SubStep 2/ get zone semantic shape file and burn it in mask directory and save
            #         it and update shapfile dataset
            polygons = [(row.geometry, CROSS_LABEL_MATRI_PROD_NAF_CHANGE[row.code])
                        if row.code in CROSS_LABEL_MATRI_PROD_NAF_CHANGE.keys() else (row.geometry, 0)
                        for idx, row in gdf.iterrows()]

            mask = features.rasterize(polygons,
                                      out_shape=out_shape,
                                      transform=out_transform,
                                      fill=0,
                                      all_touched=True)

            relative_out_f_mask = os.path.join("mask", v["zone_id"] + ".tiff")
            out_f_mask = os.path.join(OUTPUT_PATH, relative_out_f_mask)

            meta.update({"width": width,
                         "height": height,
                         "transform": out_transform,
                         "count": 1})

            with rio.open(out_f_mask, "w+", **meta) as dst:
                dst.write(np.expand_dims(mask, axis=0))

            # Substep 3/ get RVB, IRC  and compute MNH from MNS and MNT for T0 and T1 and burn it and save
            #         it in image directory and update shapfile dataset

            with rio.open(v["rvb_path"]) as src:
                rvb = src.read()
                profile = src.meta

            with rio.open(v["ir_path"]) as src:
                ir = src.read()
            # print(rvb.shape)
            # print(ir.shape)

            with rio.open(v["mns_path"]) as src:
                mns = src.read()

            # print(mns.shape)
            with rio.open(v["mnt_path"]) as src:
                mnt = src.read()

            # print(mnt.shape)
            mnh = mns - mnt
            mnh[mnh > 50.0] = 50.0
            mnh[mnh < 0.20] = 0.0
            mnh *= 5.0

            bands = np.vstack([rvb, ir, mnh])
            relative_out_f_band = os.path.join("image", v["zone_id"] + ".tiff")
            out_f_band = os.path.join(OUTPUT_PATH, relative_out_f_band)
            profile.update({"count": 5})

            with rio.open(out_f_band, "w+", **profile) as dst:
                dst.write(bands)

            l2.append({"zone_id": v["zone_id"],
                       "dep": v["dep"],
                       "year": v["year"],
                       "image": relative_out_f_band,
                       "mask": relative_out_f_mask,
                       "geometry": poly_box})

        except ValueError as e:
            print(f"zone_id {v['zone_id']}, \n error: {e} ")

        except KeyError as e:
            print(f"zone_id {v['zone_id']}, \n error: {e} ")

    out_gdf = gpd.GeoDataFrame(l2, crs=crs, geometry="geometry")
    out_gdf.to_file(os.path.join(OUTPUT_PATH, "dataset.shp"))


if __name__ == "__main__":

    run_process_1()
