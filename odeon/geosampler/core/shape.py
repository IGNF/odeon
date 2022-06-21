# Module for shapely's library manipulation / works at polygon level
from pathlib import Path
from typing import Optional, Union
from shapely.geometry import box, mapping, Point
from shapely.geometry.polygon import BaseGeometry
from shapely import wkt
import geopandas as gpd


def load_polygon_from_wkt(wkt_str: str) -> BaseGeometry:
    """

    Parameters
    ----------
    wkt_str

    Returns
    -------

    """
    try:
        return wkt.loads(wkt_str)
    except TypeError as e:
        raise e


def create_box_from_bounds(x_min: float, y_min: float, x_max: float, y_max: float) -> box:
    """

    Parameters
    ----------
    x_min
    y_min
    x_max
    y_max

    Returns
    -------
    shapely.geometry.box

    """

    return box(x_min, y_min, x_max, y_max)


def create_polygon_from_bounds(x_min: float, y_min: float, x_max: float, y_max: float) -> box:
    """

    Parameters
    ----------
    x_min
    y_min
    x_max
    y_max

    Returns
    -------
    shapely.geometry.box
    """

    return mapping(box(x_min, y_min, x_max, y_max))


def print_gdf(gdf: gpd.GeoDataFrame, filename: Union[str, Path], driver: Optional[str] = None):
    """

    Parameters
    ----------
    gdf
    filename
    driver

    Returns
    -------

    """
    params = {"filename": filename, "driver": driver} if driver is not None else {"filename": filename}
    gdf.to_file(**params)
