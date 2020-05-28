import argparse
import os
import logging
from pprint import pformat

from odeon.commons.json_interpreter import JsonInterpreter
from odeon.scripts.train import train


def parse_arguments():
    """
    Argument parsing
    """

    available_tools = ['train']

    parser = argparse.ArgumentParser()
    parser.add_argument("tool", help="command to be launched", choices=available_tools)
    parser.add_argument("-c", "--config", action='store', type=str, help="json configuration file (required)",
                        required=True)
    parser.add_argument("-v", "--verbosity", action="store_true", help="increase output verbosity", default=0)
    args = parser.parse_args()

    if args.config is None or not os.path.exists(args.config):
        logging.error("ERROR: Sampling config file not found (check path)")
        exit(1)

    try:
        with open(args.config, 'r') as json_file:
            json_dict = JsonInterpreter(json_file)
            json_dict.check_content(["data_sources", "model_setup"])

            return args.tool, json_dict.get_dict(), args.verbosity
    except IOError:
        logging.exception("JSON file incorrectly formatted")
        exit(1)

def main():

    tool, conf, verbosity = parse_arguments()

    if verbosity:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug(f"Loaded configuration: \n{pformat(conf, indent=4)}")

    if tool == "train":
        train(conf)


if __name__ == '__main__':
    main()
