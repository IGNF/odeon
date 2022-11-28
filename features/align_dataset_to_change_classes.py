# Align segmentation dataset of Gers 2016 to new change nomenclature
# Note: we can't use "serre" class, as it was not in previous nomenclature.
import os.path

import geopandas as gpd
# import numpy as np
import rasterio as rio
from tqdm import tqdm

ROOT = "/media/HP-2007S005-data/gers/supervised_dataset"
DATASET_FILE = os.path.join(ROOT, "supervised_dataset_with_stats_and_weights_2016_balanced.geojson")

cross_semantic_matrix = {0: [[0, 0, 0], "nolabel", 0, 0],
                         1: [[219, 14, 154], "batiment", 1, 1],
                         2: [[147, 142, 123], "zone_permeable", 1, 2],
                         3: [[248, 12, 0], "zone_impermeable", 1, 3],
                         4: [[31, 230, 235], "piscine", 1, 4],
                         5: [[219, 14, 154], "sol_nus", 2, 5],
                         6: [[169, 112, 1], "surface_eau", 2, 6],
                         7: [[21, 83, 174], "eau", 2, 7],
                         8: [[25, 74, 38], "coniferes", 3, 8],
                         9: [[138, 179, 160], "coupe", 3, 0],
                         10: [[70, 228, 131], "feuillus", 3, 9],
                         11: [[243, 166, 13], "broussaille", 3, 10],
                         12: [[21, 83, 174], "vigne", 3, 11],
                         13: [[255, 243, 13], "culture", 3, 12],
                         14: [[228, 223, 124], "terre_labouree", 3, 12],
                         15: [[193, 62, 236], "other", 0, 0]
                         }


def run_process():
    """
    Step 1/ load shapefile dataset
    Step 2/ add field naf_for_change_2016
    Step 3/ for each row:
        SubStep 3.1/ read raster at field naf_2016_path
        Substep 3.2/ change class based on cross semantic matrix on new array and update naf_for_change_2016 field
        Substep 3.3/ Write array as raster

    Returns
    -------

    """
    # Step 1/ load shapefile dataset
    gdf = gpd.read_file(DATASET_FILE)

    # Step 2/ add field naf_for_change_2016
    gdf["naf_for_change_2016_path"] = ""
    # print(len(gdf))

    # Step 3/ for each row:
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):

        # SubStep 3.1/ read raster at field naf_2016_path
        with rio.open(os.path.join(ROOT, row.naf_2016_path)) as src:
            img = src.read()
            profile = src.profile
        print(profile)

        # Substep 3.2/ change class based on cross semantic matrix on new array and update naf_for_change_2016 field

        for key, value in cross_semantic_matrix.items():
            img[img == key] = value[3]

        assert img.max() <= 12
        sp = row.naf_2016_path.split("naf")
        print(sp)
        gdf.loc[idx, "naf_for_change_2016_path"] = sp[0] + "naf_reduced" + sp[1]
        print(gdf.loc[idx, "naf_for_change_2016_path"])

        # Substep 3.3/ Write array as raster

        with rio.open(os.path.join(ROOT, gdf.loc[idx, "naf_for_change_2016_path"]), "w+", **profile) as dst:
            dst.write(img)


if __name__ == "__main__":

    run_process()
