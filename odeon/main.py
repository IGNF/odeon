import argparse
import os
from pprint import pformat
import pathlib
import json

from odeon.commons.timer import Timer
from odeon.commons.json_interpreter import JsonInterpreter

from odeon import LOGGER

def parse_arguments():
    """
    Argument parsing
    """

    available_tools = ['sampler_grid', 'trainer']

    parser = argparse.ArgumentParser()
    parser.add_argument("tool", help="command to be launched", choices=available_tools)
    parser.add_argument("-c", "--config", action='store', type=str, help="json configuration file (required)",
                        required=True)
    parser.add_argument("-v", "--verbosity", action="store_true", help="increase output verbosity", default=0)
    args = parser.parse_args()

    schema_path = os.path.join(pathlib.Path().absolute(), *["odeon", "scripts", "json_defaults",
                               f"{args.tool}_schema.json"])
    with open(schema_path) as schema_file:
        SCHEMA = json.load(schema_file)

    if args.config is None or not os.path.exists(args.config):
        LOGGER.error("ERROR: config file not found (check path)")
        exit(1)

    try:
        with open(args.config, 'r') as json_file:
            json_dict = JsonInterpreter(json_file)
            # json_dict.check_content(["data_sources", "model_setup"])
            if json_dict.is_valid(SCHEMA):
                return args.tool, json_dict.__dict__, args.verbosity

            # return args.tool, json_dict.get_dict(), args.verbosity
    except IOError:
        LOGGER.exception("JSON file incorrectly formatted")
        exit(1)

def main():

    tool, conf, verbosity = parse_arguments()

    if verbosity:
        LOGGER.setLevel('DEBUG')
    else:
        LOGGER.setLevel('INFO')

    LOGGER.debug(f"Loaded configuration: \n{pformat(conf, indent=4)}")

    if tool == "sampler_grid":
        from odeon.scripts.sampler_grid import grid_sample
        with Timer("Sampling"):
            image_conf, sampler_conf = conf['image'], conf['sampler']
            grid_sample(verbosity, **sampler_conf, **image_conf)
        return 0
    elif tool == "trainer":
        from odeon.scripts.trainer import train
        with Timer("Training"):
            datasource_conf = conf.get('data_source')
            model_conf = conf.get('model_setup')
            train_conf = conf.get('train_setup')
            train(verbosity, **datasource_conf, **model_conf, **train_conf)
        return 0


if __name__ == '__main__':
    main()
