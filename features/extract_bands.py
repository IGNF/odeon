import os
from typing import List

import rasterio as rio


def extract_bands(input_raster: str, output_path: str, indexes: List[int]):
    output_f = os.path.join(output_path, input_raster.split("/")[-1])
    with rio.open(input_raster) as src:
        img = src.read(indexes=indexes)
        profile = src.profile
        profile["count"] = len(indexes)
        with rio.open(output_f, "w+", **profile) as dst:
            dst.write(img)


if __name__ == "__main__":

    extract_bands("/media/HP-2007S005-data/gers/supervised_dataset/train/2019/raster/"
                  "zone_u19_504576-0000_6282496-0000.tiff",
                  "/media/HP-2007S005-home/pytorch-neural-style-transfer/data/content-images",
                  [1, 2, 3])
