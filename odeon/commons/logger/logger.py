""" OdeonLogger
Logger class in an individual module with singleton pattern: one instance


"""
import logging
import os
import datetime
from odeon.commons.folder_manager import create_folder


class ColoredFormatter(logging.Formatter):
    """
    Logging Formatter to add colors and count warning / errors
    """

    grey = "\x1b[38;21m"
    magenta = "\x1b[35;1m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "'%(asctime)s \t %(name)s  \t [%(levelname)s | ''%(filename)s:%(lineno)s] > %(message)s'"

    FORMATS = {
        logging.DEBUG: magenta + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_new_logger(name):
    """

    :param name: str name of your new logger
    :return: logging.Logger your new logger
    """

    if name in logging.root.manager.loggerDict:

        raise Exception("{} exists already".format(name))

    else:
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        return log


def get_simple_handler(level=logging.DEBUG):
    """

    Parameters
    ----------

    level logging level

    Returns
    -------

    StreamHandler

    """
    ch = logging.StreamHandler()
    ch.setLevel(level)

    return ch


def get_stream_handler(level=logging.DEBUG):
    """

    Parameters
    ----------

    level logging level

    Returns
    -------

    StreamHandler
    """

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(ColoredFormatter())
    return ch


def get_file_handler(logger: logging.Logger, dir_name, level=logging.WARNING):
    """
    :param logger:
    :param dir_name:
    :param level: min level to log in default: WARNING
    :return: logging.FileHandler
    """
    formatter = logging.Formatter('%(asctime)s \t %(name)s  \t [%(levelname)s | '
                                  '%(filename)s:%(lineno)s] > %(message)s')
    now = datetime.datetime.now()
    if not os.path.isdir(dir_name):
        create_folder(dir_name)

    fh = logging.FileHandler(
        os.path.join(dir_name, "_".join(["log", logger.name, now.strftime("%Y-%m-%d")]) + ".log")
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    return fh
