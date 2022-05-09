from os import listdir
from pathlib import Path

import pandas as pd
import rasterio


def load_geo_img_dir(img_dir: Path, img_suffix_list=[".tif", ".jp2"]) -> pd.DataFrame:
    """
    list info/metadata of images from a directory into a dataframe

    list all images which are in a img_dir and with suffix present in img_suffix_list.
    All image are open with rasterio to load their geographic metadata and all the informations
    are written in a pandas Dataframe with one row by image and with columns :

    * "name" : name (str) of the image without extension
    * "path" : full path of the image (dir+name+extension)
    * "width" : width of the image in pixel (int)
    * "height" : height of the image in pixel (int)
    * "res_x" : resolution in x axis (float)
    * "res_y" : resolution in y axis (float)
    * "ul_x" : upper left coordinate of the image in x axis (float)
    * "ul_y" : upper left coordinate of the image in y axis (float)
    * "transform" : full affine transform to convert from pixel coordinate to grund coordinate.
      follow the rasterio convention.

    Note:
        The rasterio transform convention is :

        .. code-block:: python

            transform = (
                resolution/scale x,
                rotation row/x,
                upper_left/origin x,
                rotation col/y,
                resolution/scale y,
                upper_left/origin y)

    Args:
        img_dir (Path) : Path of the directory where images will be search
        img_suffix_list  (List[str]) : list of image extension to filter files in img_dir.

    Returns:
        pd.DataFrame : Pandas dataframe with a row by images. empty if no images found

    """
    files = [
        img_dir.joinpath(f) for f in listdir(img_dir) if img_dir.joinpath(f).is_file()
    ]
    img_files = [f for f in files if f.suffix.lower() in img_suffix_list]
    img_rows = []
    for img_path in img_files:
        with rasterio.open(img_path) as ds:
            width = ds.width  # x axis
            height = ds.height  # y axis
            transform = ds.transform
            res_x = transform[0]
            res_y = transform[4]
            ul_x = transform[2]
            ul_y = transform[5]
            path = str(img_path)
            name = str(img_path.stem)
            row = {
                "name": name,
                "width": width,
                "height": height,
                "res_x": res_x,
                "res_y": res_y,
                "ul_x": ul_x,
                "ul_y": ul_y,
                "path": path,
                "transform": transform,
            }
            img_rows.append(row)
    img_df = pd.DataFrame(img_rows)
    return img_df
