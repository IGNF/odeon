import argparse
import os
import logging
from pprint import pformat

from odeon.commons.json_interpreter import JsonInterpreter
from odeon.scripts.train import train

logger = logging.getLogger(__package__)

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
        logger.error("ERROR: Sampling config file not found (check path)")
        exit(1)

    try:
        with open(args.config, 'r') as json_file:
            json_dict = JsonInterpreter(json_file)
            json_dict.check_content(["data_sources", "model_setup"])

            return args.tool, json_dict.get_dict(), args.verbosity
    except IOError:
        logger.exception("JSON file incorrectly formatted")
        exit(1)

def customize_logger():

    class CustomFormatter(logging.Formatter):
        """Logging Formatter to add colors and count warning / errors"""

        FORMATS = {
            logging.ERROR: "ERROR: %(msg)s",
            logging.WARNING: "WARNING: %(msg)s",
            logging.DEBUG: "%(asctime)s: %(levelname)s - %(message)s",
            "DEFAULT": "%(msg)s",
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    INFO_LEVELV_NUM = 19
    logging.Logger.INFOV = INFO_LEVELV_NUM
    logging.addLevelName(INFO_LEVELV_NUM, "INFOV")

    def infov(self, msg, *args, **kwargs):
        if self.isEnabledFor(INFO_LEVELV_NUM):
            self._log(INFO_LEVELV_NUM, msg, args, **kwargs)
    logging.Logger.infov = infov

    # handler
    handler = logging.StreamHandler()
    formatter = CustomFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def main():

    tool, conf, verbosity = parse_arguments()

    customize_logger()

    if verbosity:
        logger.setLevel('INFOV')
    else:
        logger.setLevel('INFO')

    logger.infov(f"Loaded configuration: \n{pformat(conf, indent=4)}")

    if tool == "train":
        train(conf)


if __name__ == '__main__':
    main()
