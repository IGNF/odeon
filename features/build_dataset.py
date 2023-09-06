import glob
import os

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import shape
from tqdm import tqdm

DIR_ROOT = "/media/store-pbf/TERR-IA/PRODUCTIONS/ANNOTATIONS/CHANTIER_D032_2016_2019_CHGT/C_SAISIES_TERMINEES"
PATTERN = "*/SEGMENTED_DATA/*.shp"
OUTPUT_PATH = "/media/DATA10T_3/gers/change_dataset"
ZONES = os.path.join(OUTPUT_PATH, "zones_vt_diff_D032_2016_2019_S1.shp")
RVB_2019_PATH = "/media/store_sdm/OCSGE/PROD/chantiers/D032_CHANGEMENT/2019/data_brutes/rvb.vrt"
IRC_2019_PATH = "/media/store_sdm/OCSGE/PROD/chantiers/D032_CHANGEMENT/2019/data_brutes/irc.vrt"
MNS_2019_PATH = "/media/store_sdm/OCSGE/PROD/chantiers/D032_CHANGEMENT/2019/data_brutes/mns.vrt"
RVB_2016_PATH = "/media/store_sdm/OCSGE/PROD/chantiers/D032_CHANGEMENT/2016/data_brutes/rvb.vrt"
IRC_2016_PATH = "/media/store_sdm/OCSGE/PROD/chantiers/D032_CHANGEMENT/2016/data_brutes/irc.vrt"
MNS_2016_PATH = "/media/store_sdm/OCSGE/PROD/chantiers/D032_CHANGEMENT/2016/data_brutes/mns.vrt"
MNT = "/media/store_sdm/OCSGE/PROD/chantiers/mnt_v2.vrt"
db_output_file = "dataset.shp"
RESAMPLING = rio.enums.Resampling.nearest
BOUNDLESS = True


def run_process_1() -> None:
    """
    Step 1/ Find shape files
    Step 2/ filter polygon with change
    Step 3/ Get Zone id for each row
    Step 4/ Create stastics by zone in zone shape file
    Step 5/ Save zone shape file and change polygons in shapefiles by zone id

    Returns
    -------

    """

    # Step 1: Find shape files

    pattern = os.path.join(DIR_ROOT, PATTERN)
    worksite_list = glob.glob(pattern)
    print(worksite_list)
    print(len(worksite_list))
    # dataset = []
    worksite_dict = {p.split("/")[-3]: p for p in worksite_list}
    print(worksite_dict)
    zone_gdf = gpd.read_file(ZONES)
    zone_gdf["area"] = zone_gdf["geometry"].apply(lambda x: x.area)
    zone_gdf["per_change"] = 0.0
    crs = None

    for key, value in worksite_dict.items():

        # Step 2/ filter polygon with change
        gdf: gpd.GeoDataFrame = gpd.read_file(value)
        crs = gdf.crs if crs is None else crs
        print(key)
        print(len(gdf))
        gdf = gdf[gdf["change"] == 1]
        print(len(gdf))

        # Step 3/ Get Zone id for each row
        if key == "UC_S1_1":
            gdf["zone_id"] = "UC_S1_1"
        else:
            gdf["zone_id"] = gdf["layer"].apply(lambda x: "_".join(
                [i for idx, i in enumerate(x.split("_")) if idx in [3, 4, 5]]))

        # Step 4/ Create stastics by zone in zone shape file
        gdf["area"] = gdf.geometry.apply(lambda x: x.area)
        for zone_id in gdf["zone_id"].unique():
            out_gdf = gdf[gdf["zone_id"] == zone_id]
            zone_change = out_gdf["area"].sum()
            zone_id_gdf = zone_gdf[zone_gdf["id_zone"] == zone_id]
            print(zone_id)
            print(len(zone_id_gdf))
            # print(zone_gdf["id_zone"].unique())
            zone_area = zone_id_gdf.iloc[0].geometry.area
            zone_gdf.loc[zone_gdf["id_zone"] == zone_id, "per_change"] = zone_change / zone_area
            # Step 5 part 1 change polygons in shapefiles by zone id
            out_gdf = gpd.GeoDataFrame(out_gdf, crs=crs, geometry="geometry")
            out_gdf.to_file(os.path.join(OUTPUT_PATH, *["zones", zone_id + ".shp"]))

    # Step 5 part 2 Save zone shape file
    zone_gdf = gpd.GeoDataFrame(zone_gdf, crs=crs, geometry="geometry")
    zone_gdf.to_file(os.path.join(OUTPUT_PATH, "zone_of_dataset.shp"))


def run_process_2() -> None:

    """
    Goal: create zone rasters and mask from database create in process 1

    Step 1/ load zone dataset
    Step 2/ Open RVB IRC MNS MNT for each date
    Step 2/ For each zone
        Substep 1/ compute transform
        SubStep 2/ get zone change shape file and burn it in change directory and save
        it and update shapfile dataset
        Substep 3/ get RVB, IRC  and compute MNH from MNS and MNT for T0 and T1 and burn it and save
        it and update shapfile dataset
    Step 4/ save shapefile dataset

    Returns
    -------

    """
    target_resolution = 0.2
    # Step 1/ load zone dataset
    zone_gdf = gpd.read_file(os.path.join(OUTPUT_PATH, "zone_of_dataset.shp"))
    crs = zone_gdf.crs
    print(zone_gdf["per_change"].mean())
    print(zone_gdf["per_change"].min())
    print(zone_gdf["per_change"].max())
    print(zone_gdf["area"].mean())
    print(zone_gdf["area"].mean())
    print(zone_gdf["area"].min())
    print(zone_gdf["area"].max())
    meta: dict = {"driver": "GTiff",
                  "crs": crs,
                  "TILED": "YES",
                  "dtype": "uint8"}
    meta_img: dict = meta.copy()
    meta_img.update({"COMPRESS": "DEFLATE"})

    # Step 2/ Open RVB IRC MNS MNT for each date

    rvb_2016 = rio.open(RVB_2016_PATH)
    rvb_2019 = rio.open(RVB_2019_PATH)
    irc_2016 = rio.open(IRC_2019_PATH)
    irc_2019 = rio.open(IRC_2019_PATH)
    mns_2016 = rio.open(MNS_2016_PATH)
    mns_2019 = rio.open(MNS_2019_PATH)
    mnt = rio.open(MNT)

    # Step 2/ For each zone
    for idx, row in tqdm(zone_gdf.iterrows(), total=len(zone_gdf)):
        # print(row)

        # Substep 1/ compute transform

        bounds = row.geometry.bounds
        x_min, y_min, x_max, y_max = bounds[0], bounds[1], bounds[2], bounds[3]
        width = int((x_max - x_min) / target_resolution)
        height = int((y_max - y_min) / target_resolution)
        out_shape = (height, width)
        out_transform = from_bounds(west=x_min, south=y_min, east=x_max, north=y_max, width=width, height=height)
        # print(out_transform)

        # SubStep 2/ get zone change shape file and burn it in change directory and save
        #         it and update shapfile dataset

        zone_id = row.id_zone
        # print(zone_id)
        f = os.path.join(OUTPUT_PATH, *["zones", zone_id + ".shp"])
        if os.path.isfile(f):
            zone_id_gdf = gpd.read_file(f)
            polygons = [(row.geometry, 1) for idx, row in zone_id_gdf.iterrows()]
            mask = features.rasterize(polygons,
                                      out_shape=out_shape,
                                      transform=out_transform,
                                      fill=0,
                                      all_touched=True)
        else:
            mask = np.zeros((width, height), dtype=np.uint8)

        relative_change_path = os.path.join("mask", zone_id + ".tiff")
        out_f = os.path.join(OUTPUT_PATH, relative_change_path)

        meta.update({"width": width,
                     "height": height,
                     "transform": out_transform,
                     "count": 1})
        with rio.open(out_f, "w+", **meta) as dst:
            dst.write(np.expand_dims(mask, axis=0))
            zone_gdf.loc[idx, "change_pat"] = relative_change_path
        # Substep 3/ get RVB, IRC  and compute MNH from MNS and MNT for T0 and T1 and burn it and save
        #         it and update shapfile dataset

        # T0
        rvb_t0 = rvb_2016.read(window=window_from_bounds(x_min, y_min, x_max, y_max, rvb_2016.meta["transform"]),
                               out_shape=(3, height, width),
                               resampling=RESAMPLING,
                               boundless=BOUNDLESS)

        irc_t0 = irc_2016.read(window=window_from_bounds(x_min, y_min, x_max, y_max, irc_2016.meta["transform"]),
                               out_shape=(3, height, width),
                               resampling=RESAMPLING,
                               boundless=BOUNDLESS)

        mns_t0 = mns_2016.read(window=window_from_bounds(x_min, y_min, x_max, y_max, mns_2016.meta["transform"]),
                               out_shape=(1, height, width),
                               resampling=RESAMPLING,
                               boundless=BOUNDLESS)

        mnt_t = mnt.read(window=window_from_bounds(x_min, y_min, x_max, y_max, mnt.meta["transform"]),
                         out_shape=(1, height, width),
                         resampling=RESAMPLING,
                         boundless=BOUNDLESS)

        mnh_t0 = mns_t0 - mnt_t
        mnh_t0[mnh_t0 > 50.0] = 50.0
        mnh_t0[mnh_t0 < 0.20] = 0.0
        mnh_t0 *= 5.0
        bands_t0 = np.vstack([rvb_t0, np.expand_dims(irc_t0[0], axis=0), mnh_t0])
        # print(f"bands t0 shape: {bands_t0.shape}")
        meta_img.update({"width": width,
                         "height": height,
                         "transform": out_transform,
                         "count": 5})
        relative_path = os.path.join("T0", zone_id + ".tiff")
        img_t0_f = os.path.join(OUTPUT_PATH, relative_path)
        zone_gdf.loc[idx, "T0_path"] = relative_path
        with rio.open(img_t0_f, "w+", **meta_img) as dst:
            dst.write(bands_t0)

        # T1
        rvb_t1 = rvb_2019.read(window=window_from_bounds(x_min, y_min, x_max, y_max, rvb_2019.meta["transform"]),
                               out_shape=(3, height, width),
                               resampling=RESAMPLING,
                               boundless=BOUNDLESS)

        irc_t1 = irc_2019.read(window=window_from_bounds(x_min, y_min, x_max, y_max, irc_2019.meta["transform"]),
                               out_shape=(3, height, width),
                               resampling=RESAMPLING,
                               boundless=BOUNDLESS)

        mns_t1 = mns_2019.read(window=window_from_bounds(x_min, y_min, x_max, y_max, mns_2019.meta["transform"]),
                               out_shape=(1, height, width),
                               resampling=RESAMPLING,
                               boundless=BOUNDLESS)

        mnh_t1 = mns_t1 - mnt_t
        mnh_t1[mnh_t1 > 50.0] = 50.0
        mnh_t1[mnh_t1 < 0.20] = 0.0
        mnh_t1 *= 5.0
        bands_t1 = np.vstack([rvb_t1, np.expand_dims(irc_t1[0], axis=0), mnh_t1])
        # print(f"bands t0 shape: {bands_t0.shape}")
        meta_img.update({"width": width,
                         "height": height,
                         "transform": out_transform,
                         "count": 5})
        relative_path = os.path.join("T1", zone_id + ".tiff")
        img_t1_f = os.path.join(OUTPUT_PATH, relative_path)
        zone_gdf.loc[idx, "t1_path"] = relative_path
        with rio.open(img_t1_f, "w+", **meta_img) as dst:
            dst.write(bands_t1)

    # Step 4/ save shapefile dataset
    zone_gdf.to_file(os.path.join(OUTPUT_PATH, "dataset_v1.shp"))


def run_process_3():
    """
    Goal: vectorize change from raster mask and save them in GeoDataFrame

    Step 1/ load zone dataset
    Step 2/ For each zone
        Substep 1/ read mask
        Substep 2/ vectorize mask and create GeoSeries as dict
    Step 3/ Save GeoDataFrame

    Returns
    -------

    """

    # Step 1/ load zone dataset

    gdf = gpd.read_file(os.path.join(OUTPUT_PATH, "dataset_v1.shp"))
    crs = gdf.crs
    d = []  # list to stack GeoSeries as dict
    # Step 2/ For each zone
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):

        # Substep 1/ read mask
        change_path = os.path.join(OUTPUT_PATH, row.change_pat)
        with rio.open(change_path) as src:
            mask = src.read().astype("uint8")
            transform = src.meta["transform"]

        # Substep 2/ vectorize mask and create GeoSeries as dict
        if mask.sum() > 0:
            for idx, (polygon, value) in enumerate(rio.features.shapes(mask, mask=mask, transform=transform)):
                uuid = row.id_zone + "-" + str(idx)
                d_row = {"geometry": shape(polygon), "change": value, "uid": uuid, "zone_id": row.id_zone}
                d.append(d_row)

    # Step 3/ Save GeoDataFrame
    out_gdf = gpd.GeoDataFrame(d, geometry="geometry", crs=crs)
    out_gdf.to_file(os.path.join(OUTPUT_PATH, "change_vectorized.shp"))


def run_process_4():
    """

        Goal: vectorize change from raster mask and save them in GeoDataFrame

        Step 1/ load zone dataset
        Step 2 / compute stats
        Step 3 /
        Step 3/ Save GeoDataFrame

    """

    gdf = gpd.read_file(os.path.join(OUTPUT_PATH, "zone_of_dataset.shp"))
    print(len(gdf))
    print(f"nombre de zones: {len(gdf)}")
    print(f"surface moyenne des zones: {gdf['area'].mean()}")
    print(f"nombre de zones sans changement: {len(gdf[gdf['per_change'] == 0.0])}")
    print(f" surface totale des zones: {gdf['area'].sum()}")
    print(f"zone avec le pourcentage minimum de changement: {gdf['per_change'].min()}")
    print(f"zone avec le pourcentage maximum de changement: {gdf['per_change'].max()}")
    print(f"pourcentage moyen de changement: {gdf['per_change'].mean()}")
    print(f"écart type du pourcentage de changement par zone: {gdf['per_change'].std()}")
    print(f"histogram du pourcentage de changement par zone: {gdf['per_change'].hist(bins=10)}")
    ax = gdf['per_change'].hist(bins=10)
    fig = ax.get_figure()
    fig.savefig(os.path.join(OUTPUT_PATH, 'figure.png'))


if __name__ == "__main__":
    # run_process_1()
    # run_process_2()
    # run_process_3()
    run_process_4()
