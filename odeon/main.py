import argparse
import os
import sys
from pprint import pformat
import pathlib
import json
from odeon.commons.timer import Timer
from odeon.commons.json_interpreter import JsonInterpreter
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes


def parse_arguments():

    """
    parse arguments
    Returns
    -------

    """

    available_tools = ['sample_grid', 'train', 'generate']

    parser = argparse.ArgumentParser()
    parser.add_argument("tool", help="command to be launched", choices=available_tools)
    parser.add_argument("-c", "--config", action='store', type=str, help="json configuration file (required)",
                        required=True)
    parser.add_argument("-v", "--verbosity", action="store_true", help="increase output verbosity", default=0)
    args = parser.parse_args()
    schema_path = os.path.join(pathlib.Path().absolute(), *["odeon", "scripts", "json_defaults",
                               f"{args.tool}_schema.json"])

    with open(schema_path) as schema_file:
        schema = json.load(schema_file)

    if args.config is None or not os.path.exists(args.config):

        message = "ERROR: config file not found (check path)"
        LOGGER.error(message)
        raise OdeonError(ErrorCodes.ERR_IO, message)

    try:
        with open(args.config, 'r') as json_file:
            json_dict = JsonInterpreter(json_file)
            # json_dict.check_content(["data_sources", "model_setup"])
            if json_dict.is_valid(schema):
                return args.tool, json_dict.__dict__, args.verbosity

            # return args.tool, json_dict.get_dict(), args.verbosity
    except IOError:

        message = "JSON file incorrectly formatted"
        LOGGER.error(message + str(IOError))

        raise OdeonError(
            ErrorCodes.ERR_JSON_SCHEMA_ERROR,
            message,
            stack_trace=IOError)


def main():

    try:

        tool, conf, verbosity = parse_arguments()

    except OdeonError:

        return ErrorCodes.ERR_MAIN_CONF_ERROR

    if verbosity:

        LOGGER.setLevel('DEBUG')
    else:
        LOGGER.setLevel('INFO')

    LOGGER.debug(f"Loaded configuration: \n{pformat(conf, indent=4)}")

    if tool == "sample_grid":

        from odeon.scripts.sample_grid import grid_sample

        with Timer("Sample Grid"):

            image_conf, sampler_conf = conf['image'], conf['sampler']
            grid_sample(verbosity, **sampler_conf, **image_conf)

        return 0

    elif tool == "train":

        from odeon.scripts.train import train

        with Timer("Training"):

            datasource_conf = conf.get('data_source')
            model_conf = conf.get('model_setup')
            train_conf = conf.get('train_setup')
            train(verbosity, **datasource_conf, **model_conf, **train_conf)

        return 0

    elif tool == "generate":

        from odeon.scripts.generate import generate

        with Timer("Generate data"):

            try:
                image_layers = conf["image_layers"]
                vector_classes = conf["vector_classes"]
                image = conf["image"]
                generator = conf["generator"]

                generate(image_layers,
                         vector_classes,
                         **image,
                         **generator)
                return 0

            except OdeonError as oe:

                LOGGER.error(oe)
                return oe.error_code

    else:
        return ErrorCodes.ERR_MAIN_CONF_ERROR


if __name__ == '__main__':

    sys.exit(main())
