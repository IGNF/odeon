import os
import geopandas as gpd
import rasterio as rio
from rasterio.windows import from_bounds, transform
from tqdm import tqdm
import numpy as np

tests_zones = ["v15", "v5", "v11", "u4", "u8", "u10", "n2", "n14", "n18"]
# ROOT = "/media/DATA10T_3"
ROOT = "/media/hd"
# ROOT_OUT = "/media/DATA10T_3"
ROOT_OUT = "/media/hd"
# NAS = "/media/NAS"
GERS_DATA_PATH = os.path.join(ROOT, "gers")
RVB_VRT_2019 = os.path.join(GERS_DATA_PATH, "rvb-2019.vrt")
RVB_VRT_2016 = os.path.join(GERS_DATA_PATH, "rvb-2016.vrt")
IRC_VRT_2019 = os.path.join(GERS_DATA_PATH, "irc-2019.vrt")
IRC_VRT_2016 = os.path.join(GERS_DATA_PATH, "irc-2016.vrt")
PATH_TO_UNSUPERVISED_DB = os.path.join(GERS_DATA_PATH, "32_256X256_unsupervised.geojson")
OUTPUT_PATH = os.path.join(GERS_DATA_PATH, "unsupervised_db")
OUTPUT_SHAPE = (3, 1024, 1024)


def check_unsupervised_db():
    pass


def build_unsupervised_dataset():

    gdf = gpd.read_file(PATH_TO_UNSUPERVISED_DB)

    with rio.Env(GDAL_MAX_DATASET_POOL_SIZE=5) as env:

        with rio.open(RVB_VRT_2019) as src_2019:
            with rio.open(RVB_VRT_2016) as src_2016:

                src_meta_2019 = src_2019.meta.copy()
                src_transform_2019 = src_meta_2019["transform"]
                src_meta_2016 = src_2016.meta.copy()
                src_transform_2016 = src_meta_2016["transform"]
                # print(src_transform)
                meta = src_meta_2019.copy()
                meta["COMPRESS"] = "JPEG"
                # meta["ZLEVEL"] = 1
                meta["TILED"] = True
                meta["driver"] = "GTiff"
                meta["JPEG_QUALITY"] = 95
                meta["PHOTOMETRIC"] = "YCBCR"
                meta["count"] = 3
                meta["width"] = 1024
                meta["height"] = 1024
                meta_2016 = meta.copy()

                for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):

                            bounds = row.geometry.bounds
                            # print(bounds)
                            window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src_transform_2019)
                            dst_transform = transform(window, src_transform_2019)
                            meta["transform"] = dst_transform
                            bands = src_2019.read(window=window, out_shape=(3, 1024, 1024))
                            # bands = np.zeros((3, 1024, 1024)).astype("uint8")
                            # print(bands.shape)
                            x = "{:.4f}".format(bounds[0]).replace(".", "-")
                            y = "{:.4f}".format(bounds[3]).replace(".", "-")
                            filename = f"rvb_2019_{x}_{y}.tif"
                            relative_output_file = os.path.join("2019/RVB", filename)
                            absolute_output_path = os.path.join(OUTPUT_PATH, relative_output_file)

                            with rio.open(absolute_output_path, "w+", **meta) as dst:

                                dst.write(bands)

                            gdf.loc[idx, "rvb_2019"] = relative_output_file

                            window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src_transform_2016)
                            dst_transform = transform(window, src_transform_2016)
                            meta_2016["transform"] = dst_transform
                            bands = src_2016.read(window=window, out_shape=(3, 1024, 1024))
                            # bands = np.zeros((3, 1024, 1024)).astype("uint8")
                            # print(bands.shape)
                            x = "{:.4f}".format(bounds[0]).replace(".", "-")
                            y = "{:.4f}".format(bounds[3]).replace(".", "-")
                            filename = f"rvb_2016_{x}_{y}.tif"
                            relative_output_file = os.path.join("2016/RVB", filename)
                            absolute_output_path = os.path.join(OUTPUT_PATH, relative_output_file)

                            with rio.open(absolute_output_path, "w+", **meta_2016) as dst:

                                dst.write(bands)

                            gdf.loc[idx, "rvb_2016"] = relative_output_file

    gdf.to_file(os.path.join(OUTPUT_PATH, "unsupervised_dataset.geojson"), driver="GeoJSON")


if __name__ == "__main__":

    build_unsupervised_dataset()
# toto12