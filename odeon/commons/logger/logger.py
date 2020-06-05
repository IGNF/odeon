import logging
import os
import datetime

""" OdeonLogger
Logger class in an individual module to be called once and only once

"""


class OdeonLogger(object):
    """
    Logger class to log event in the Odeon App
    ...

    Attributes
    ----------
    _logger : logger object

        list of (parameters, values).

    Methods
    -------
    get_logger()
        return the logging instance

    """
    _logger = None

    def __init__(self):

        self._logger = logging.getLogger("crumbs")
        self._logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')

        now = datetime.datetime.now()
        dir_name = "./log"

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        file_handler = logging.FileHandler(dir_name + "/log_" + now.strftime("%Y-%m-%d")+".log")

        stream_handler = logging.StreamHandler()

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(stream_handler)

    def get_logger(self):
        return self._logger
