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
import sys
from typing import Tuple
from commons.json_interpreter import JsonInterpreter
from commons.timer import Timer
from sampler.sampler_grid import grid_sample
import json
import pathlib
from commons.logger.logger import get_new_logger, get_stream_handler

try:
    print("name {}".format(__name__))
    LOGGER = get_new_logger(__name__)
    LOGGER.addHandler(get_stream_handler())
except Exception as e:
    print(e)
    raise e

SCHEMA_PATH = os.path.join(pathlib.Path().absolute(), *["sampler", "sampler_schema.json"])
with open(SCHEMA_PATH) as json_file:
    SAMPLER_SCHEMA = json.load(json_file)


def main() -> int:
    with Timer("Sampling"):

        image_conf, sampler_conf, verbosity = parse_arguments()
        LOGGER.info("Sampling started")
        print("sampling started")
        # if image_conf is not None and sampler_conf is not None:  # TODO simplify
        grid_sample(verbosity, **sampler_conf, **image_conf)
    return 0


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
        sys.exit(1)

    try:
        with open(args.config, 'r') as json_file:
            json_dict = JsonInterpreter(json_file)
            # json_dict.check_content(["image", "sampler"])
            if json_dict.is_valid(SAMPLER_SCHEMA):
                return json_dict.get_image(), json_dict.get_sampler(), args.verbosity
            else:
                LOGGER.fatal("the sampling has stopped due to a bad json configuration file")
                sys.exit(1)
    except IOError as ioe:
        LOGGER.error("JSON file incorrectly formatted \n detail {}".format(str(ioe)))
        sys.exit(1)
    except TypeError as te:
        LOGGER.error("You have probably wrong arguments to JsonInterpreter \n detail {}".format(str(te)))
        sys.exit(1)
    exce


if __name__ == "__main__":
    sys.exit(main())
