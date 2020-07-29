import fiona
from tqdm import tqdm
from odeon import LOGGER


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
