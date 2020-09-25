import fiona
from tqdm import tqdm
from odeon import LOGGER
import geopandas as gpd
from shapely.geometry import box, mapping


def stack_shape(shape_file, value=1):
    """
    Preprocess of stacking shapes in a list of tuple as entry of rasterio
    rasterize function

    Parameters
    ----------

    shape_file : str
     the shape file path where
    value
     an integer to associate with each shape before rasterization

    Returns
    -------

    a list of tuples (shape, value) in a a rasterio format before rasterization

    Raises
    ------

    fiona.errors.DataIOError
    fiona.errors.DriverIOError

    """
    try:

        with fiona.open(shape_file) as shapes:

            tuple_shapes = []

            for shape in tqdm(shapes):

                tuple_shapes.append([shape['geometry'], value])

        return tuple_shapes

    except fiona.errors.DataIOError as fed:

        msg = f"an exception during the opening of vectore file {shape_file}," \
              f" a driver or a path problem? detail: {fed}"
        LOGGER.warning(msg)
        raise fiona.errors.DataIOError(msg)

    except fiona.errors.DriverIOError as fedio:

        msg = f"an exception during the opening of vectore file {shape_file}," \
              f" a driver or a path problem? detail: {fedio}"
        LOGGER.warning(msg)
        raise fiona.errors.DataIOError(msg)


def get_crs_from_shapefile(shp):
    """
    helper function to avoir warning with shapefile made with
    the old standard format of esri shapefile, still used by qgis, ogr.
    """

    if "init" in shp.crs.keys():

        return shp.crs["init"]

    else:

        return shp.crs


def build_geo_data_frame_from_shape_file(shape_file):
    """

    Parameters
    ----------
    shape_file

    Returns
    -------
    GeoDataFrame

    """

    try:

        with fiona.open(shape_file) as shp:

            return gpd.GeoDataFrame.from_features(shp, crs=get_crs_from_shapefile(shp))

    except fiona._err.CPLE_AppDefinedError as error:

        LOGGER.error(error)


def build_geo_data_frame_from_array(array, crs):
    """

    Parameters
    ----------
    array : list
    crs : crs in compatible fiona format

    Returns
    -------
    GeoDataFrame

    """
    min_x, min_y, max_x, max_y = array[0], array[1], array[2], array[3]
    shape_list = [{"id": 1, "geometry": box(min_x, max_y, max_x, min_y)}]
    return gpd.GeoDataFrame(shape_list, crs=crs, geometry="geometry")


def create_box_from_bounds(x_min, x_max, y_min, y_max):

    return box(x_min, y_max, x_max, y_min)


def create_polygon_from_bounds(x_min, x_max, y_min, y_max):

    return mapping(box(x_min, y_max, x_max, y_min))
