"""Grid sampling

This module performs a grid sampling from a file containing (multi)polygons. This is the first step in the ODEON
suite.

From a single shapefile containing several (multi)polygons, this code generates as many csv files as features.

Each csv file will contain the (x,y) coordinates in the same coordinate reference system as the input shapefile.
The distance between the points are computed in order to extract images with the dimension expressed in the json file
without overlap.


Example
-------
    Call this module from the root of the project:

    $ python -m src.grid_sampling -c src/json/grid_sampling.json -v

    This will read the configuration from a json file and create as many csv files as there are polygons in the
    shapefile.


Notes
-----
    * [Todo] implement default values for "image_size_pixel" and "pixel_size_meter_per_pixel" so they can be
    skipped in json (see json_interpreter)

"""

import argparse
import os
from sys import exit
from typing import Tuple, List, Dict
import fiona
from numpy import linspace
from shapely.geometry import shape, box, mapping, Point
from tqdm import tqdm
import commons.folder_manager as fm
from commons.json_interpreter import JsonInterpreter
from commons.timer import Timer
from commons.logger.logger import OdeonLogger

SAMPLER_SCHEMA = {
    "type": "object",
    "properties": {"image": {"type": "object",
                             "properties":
                             {"image_size_pixel": {"type": "integer", "default": 256},
                              "pixel_size_meter_per_pixel": {"type": "number", "default": 0.2}
                              }
                             },
                   "sampler": {"type": "object",
                               "properties":
                               {"input_file":
                                {"type": "string",
                                 "default": "/media/ssd/datasets/ocsge/odeon_data/learning_zones/zone_33_1_crs.shp"},
                                "output_pattern":
                                    {"type": "string",
                                     "default": "/media/ssd/datasets/ocsge/odeon_data/data/odeon_sample/zone_33_1.csv"},
                                "shift": {"enum": [0, 1], "default": 0}
                                }
                               },
                   "required": ["image", "sampler"]
                   }
}

LOGGER = OdeonLogger().get_logger()


def main() -> None:
    with Timer("Sampling"):

        image_conf, sampler_conf, verbosity = parse_arguments()
        LOGGER.info("Sampling started")
        # if image_conf is not None and sampler_conf is not None:  # TODO simplify
        grid_sample(verbosity, **sampler_conf, **image_conf)


def parse_arguments() -> Tuple:
    """
    Argument parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action='store', type=str, help="json configuration file (required)",
                        required=True)
    parser.add_argument("-v", "--verbosity", action="store_true", help="increase output verbosity", default=0)
    args = parser.parse_args()

    if args.config is None or not os.path.exists(args.config):
        LOGGER.error("ERROR: Sampling config file not found (check path)")
        exit(1)

    try:
        with open(args.config, 'r') as json_file:
            json_dict = JsonInterpreter(json_file, LOGGER)
            # json_dict.check_content(["image", "sampler"])
            if json_dict.is_valid(SAMPLER_SCHEMA):
                return json_dict.get_image(), json_dict.get_sampler(), args.verbosity
            else:
                LOGGER.fatal("the sampling has stopped due to a bad json configuration file")
                exit(1)
    except IOError as ioe:
        LOGGER.error("JSON file incorrectly formatted \n detail {}".format(str(ioe)))
        exit(1)


def setup_output(output_pattern):
    """
    Create the folder for the files created during the sampling
    :param output_pattern: the pattern to follow for the files
    """
    path = os.path.split(output_pattern)
    fm.create_folder(path)


def get_geometries_from_shp(shapefile) -> Tuple[List, Dict]:
    """
    Returns a list of geometries ordered by x, then y
    Parameters
    ----------
    shapefile : str
        shapefile containing polygons (usually within a "departement")

    Returns
    -------
    list_of_geometry : list
        list of geometries (Shapely Polygons),
    crs : dict
        coordinates system used in the shapefile

    """
    geometry_list = []
    with fiona.open(shapefile, 'r') as layer:

        crs = layer.crs
        for feature in tqdm(layer):
            geometry = shape(feature['geometry'])
            geometry_list.append(geometry)

    return geometry_list, crs


def generate_filename(output_pattern, no_of_samples) -> List:
    """
    Returns a list of filenames with complete path
    Parameters
    ----------
    output_pattern : str
        the pattern to follow when generating filenames
    no_of_samples : int
        number of filename to generate

    Returns
    -------
    filename_list : list
        a list of filenames
    """
    filename_list = []
    path, pattern = os.path.split(output_pattern)
    if "*" not in pattern:
        pattern = "*_" + pattern

    for i in tqdm(range(no_of_samples)):
        filename = os.path.join(path, pattern.replace("*", "zone" + str(i + 1)))
        filename_list.append(filename)

    return filename_list


def generate_csv(geometry_list, filename_list, side, crs, strict_inclusion, shift, verbose):
    """
    Generate a list of coordinates (x,y) for each geometry and save this list
    Parameters
    ----------
    geometry_list : list
        list of geometry (square shaped areas)
    filename_list : list
        list of filename that will
    side : float
        size of a square image in meter
    crs : dict
        coordinate system
    strict_inclusion : boolean
        True if the tile must be strictly included in the polygonal shape
    shift : boolean
        True to shift samples by half the size of a tile
    verbose : bool
        verbose level
    """
    geometry_list.sort(key=lambda g: (g.bounds[0], g.bounds[1]))
    bounding_box_list = [geometry.bounds for geometry in geometry_list]  # bbox: (x1, y1 (SW Point) x2, y2 (NE Point))
    # TODO ? bounding_box_list.sort(key=lambda x: (x[0], x[1]))  # sort by x1 then y2
    data = zip(filename_list, bounding_box_list, geometry_list)
    for filename, bounding_box, geometry in tqdm(data):
        # find limits
        x1, y1, x2, y2 = bounding_box
        if shift:
            x1, y1, x2, y2 = x1 + side / 2, y1 + side / 2, x2 - side / 2, y2 - side / 2
        x_num, y_num = (x2 - x1) / side, (y2 - y1) / side  # number of samples

        coordinates = [[(x, y) for x in linspace(x1 + side / 2, x2 - side / 2, int(x_num))] for y in
                       linspace(y1 + side / 2, y2 - side / 2, int(y_num))]

        coordinates = [j for sub in coordinates for j in sub]  # transform into a list of tuple (x, y)

        if strict_inclusion:
            coordinates = [c for c in coordinates if included(c[0], c[1], side, geometry)]
        save_output(coordinates, filename, side, crs, verbose)
        if verbose:
            LOGGER.debug(f"[{filename}]: {len(coordinates)} points")


def included(x, y, side, geometry):
    tile = box(x - side / 2, y - side / 2, x + side / 2, y + side / 2)
    return tile.intersection(geometry).area / (side * side) > 0.999


def save_output(coordinates, filename, side, crs, verbose):
    """
    Save coordinates and, if verbose is on, save shapefiles (area and center) for visual
    inspection
    Parameters
    ----------
    coordinates : list
        list of coordinates
    filename : str
        output file containing the coordinate in csv format
    side : float
        size of an image
    crs : dict
        coordinate system
    verbose : bool
        verbose level
    """
    # Save into a csv file
    csv_file = open(filename, 'w', encoding='utf-8', errors='ignore')
    for (x, y) in coordinates:
        csv_file.write(f"{round(x, 4)}; {round(y, 4)}\n")
    if verbose:
        # Save the patch shape into a shp file
        shp_filename = filename[:-4] + "_area.shp"
        shp_schema = {'geometry': 'Polygon', 'properties': {'id_sample': 'int'}}
        shp_file = fiona.open(shp_filename, 'w', crs=crs, driver='ESRI Shapefile', schema=shp_schema)
        for i, (x, y) in enumerate(coordinates):
            dx = side / 2
            shp_file.write({
                'properties': {'id_sample': i},
                'geometry': mapping(box(x - dx, y - dx, x + dx, y + dx))
            })
        # Save the patch center into a shp file
        shp_filename = filename[:-4] + "_center.shp"
        shp_schema = {'geometry': 'Point', 'properties': {'id_sample': 'int'}}
        shp_file = fiona.open(shp_filename, 'w', crs=crs, driver='ESRI Shapefile', schema=shp_schema)
        for i, (x, y) in enumerate(coordinates):
            shp_file.write({
                'properties': {'id_sample': i},
                'geometry': mapping(Point(float(x), float(y)))
            })


def grid_sample(verbose, input_file, output_pattern, image_size_pixel, pixel_size_meter_per_pixel,
                strict_inclusion=False, shift=False):
    """

    Parameters
    ----------
    verbose : bool
        verbose level
    input_file : str
        shapefile containing the zones of interest for one "departement"
    output_pattern : str
        filename pattern used for each zone
    image_size_pixel : int
        number of pixel of an images
    pixel_size_meter_per_pixel : float
        meter per pixel
    strict_inclusion : boolean
        True if the tile must be strictly included in the polygonal shape
    shift : boolean
        True to shift samples by half the size of a tile
    """
    if verbose:
        LOGGER.debug("Configuration :")
        LOGGER.debug(f"\tinput shapefile: {input_file}")
        LOGGER.debug(f"\toutput pattern: {output_pattern}")
        LOGGER.debug(f"\timage size (pixel): {image_size_pixel}")
        LOGGER.debug(f"\tpixel size (meter per pixel): {pixel_size_meter_per_pixel}")
        LOGGER.debug(f"\tstrict_inclusion: {strict_inclusion}")
        LOGGER.debug(f"\tshift (1 to shift centers): {shift}")

    if not os.path.isfile(input_file):
        LOGGER.debug(f"ERROR: file nor found: {input_file}")
    fm.create_folder(os.path.dirname(output_pattern))

    # read shape file and count shapes, order them by x
    LOGGER.info("retrieve geometry")
    geometry_list, crs = get_geometries_from_shp(input_file)
    # build output filename and create if necessary the output folder
    LOGGER.info("generate filename")
    filename_list = generate_filename(output_pattern, len(geometry_list))
    # get side of the expected images
    side = image_size_pixel * pixel_size_meter_per_pixel
    # generate csv
    LOGGER.info("generate csv")
    generate_csv(geometry_list, filename_list, side, crs, strict_inclusion, shift, verbose)

    return


if __name__ == "__main__":
    main()
