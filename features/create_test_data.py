import os
import pathlib
import shutil

import geopandas as gpd


def create_test_patch_data(database: str, output_dir: str, n_sample: int = 3) -> None:
    gdf = gpd.read_file(database)
    gdf = gdf.head(n=n_sample)

    for idx, row in gdf.iterrows():

        pathlib.Path(os.path.join(output_dir, os.path.dirname(row["raster_2019_path"]))).mkdir(parents=True,
                                                                                               exist_ok=True)
        pathlib.Path(os.path.join(output_dir, os.path.dirname(row["naf_2019_path"]))).mkdir(parents=True,
                                                                                            exist_ok=True)
        pathlib.Path(os.path.join(output_dir, os.path.dirname(row["urbain_2019_path"]))).mkdir(parents=True,
                                                                                               exist_ok=True)
        pathlib.Path(os.path.join(output_dir, os.path.dirname(row["raster_2016_path"]))).mkdir(parents=True,
                                                                                               exist_ok=True)
        pathlib.Path(os.path.join(output_dir, os.path.dirname(row["naf_2016_path"]))).mkdir(parents=True,
                                                                                            exist_ok=True)
        pathlib.Path(os.path.join(output_dir, os.path.dirname(row["urbain_2016_path"]))).mkdir(parents=True,
                                                                                               exist_ok=True)
        shutil.copy(os.path.join(os.path.dirname(database), row["raster_2019_path"]), os.path.join(
            output_dir, row["raster_2019_path"]))
        shutil.copy(os.path.join(os.path.dirname(database), row["naf_2019_path"]), os.path.join(
            output_dir, row["naf_2019_path"]))
        shutil.copy(os.path.join(os.path.dirname(database), row["urbain_2019_path"]), os.path.join(
            output_dir, row["urbain_2019_path"]))
        shutil.copy(os.path.join(os.path.dirname(database), row["raster_2016_path"]), os.path.join(
            output_dir, row["raster_2016_path"]))
        shutil.copy(os.path.join(os.path.dirname(database), row["naf_2016_path"]), os.path.join(
            output_dir, row["naf_2016_path"]))
        shutil.copy(os.path.join(os.path.dirname(database), row["urbain_2016_path"]), os.path.join(
            output_dir, row["urbain_2016_path"]))
        gdf.to_file(os.path.join(output_dir, "test_patch_data.geojson"), driver="GeoJSON")


def create_test_zone_data(database: str, output_dir: str, n_sample: int = 3) -> None:

    gdf = gpd.read_file(database)
    gdf = gdf.head(n=n_sample)

    for idx, row in gdf.iterrows():

        pathlib.Path(os.path.join(output_dir, os.path.dirname(row["change_pat"]))).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(output_dir, os.path.dirname(row["T0_path"]))).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(output_dir, os.path.dirname(row["t1_path"]))).mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(os.path.dirname(
            database), row["change_pat"]), os.path.join(output_dir, row["change_pat"]))
        shutil.copy(os.path.join(os.path.dirname(database), row["T0_path"]), os.path.join(output_dir, row["T0_path"]))
        shutil.copy(os.path.join(os.path.dirname(database), row["t1_path"]), os.path.join(output_dir, row["t1_path"]))
        gdf.to_file(os.path.join(output_dir, "test_zone_data.shp"))


if __name__ == "__main__":

    create_test_patch_data(
        "/media/HP-2007S005-data/gers/supervised_dataset/supervised_dataset_with_stats_and_weights.geojson",
        "/home/ign.fr/skhelifi/dev/odeon/tests/test_data")
