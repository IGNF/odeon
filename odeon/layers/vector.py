# Module for shapely's library manipulation / works at polygon level
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
from shapely import wkt
from shapely.geometry import box, mapping
from shapely.geometry.polygon import BaseGeometry

from .core.types import BOUNDS


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


def create_gdf_from_list(polygons: List, crs, geometry_column="geometry") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(data=polygons, crs=crs, geometry=geometry_column)


def spatial_join_within_bounds(gdf, bounds: BOUNDS, how='inner', predicate='intersects') -> list[dict]:
    """
    Perform a spatial join between a GeoDataFrame and a bounding box.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame to be joined.
    bounds : list
        A list of four coordinates representing the bounding box in the
        form [minx, miny, maxx, maxy].
    how : str, optional
        The type of join to be performed. Default is 'inner'.
        Options include 'left', 'right', 'inner', 'outer'.
    predicate : str, optional
        The type of geometric operation to use in the join. Default is 'intersects'.
        Options include 'contains', 'within', 'intersects', 'touches', 'crosses',
        'overlaps'.

    Returns
    -------
    geopandas.GeoDataFrame
        The resulting GeoDataFrame after the spatial join.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> data = {'geometry': [Point(1, 1), Point(2, 2), Point(3, 3)]}
    >>> gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    >>> bounds = [0, 0, 2, 2]
    >>> result = spatial_join_within_bounds(gdf, bounds)
    >>> result
       geometry
    0  POINT (1.00000 1.00000)
    1  POINT (2.00000 2.00000)
    """

    x_min, y_min, x_max, y_max = bounds
    bounding_box = create_polygon_from_bounds(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    bbox_gdf = gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[bounding_box])
    result = gpd.sjoin(gdf, bbox_gdf, how=how, predicate=predicate).to_dict(orient='records')
    return list(result)
