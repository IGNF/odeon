"""Logging helper module"""
import logging

DEFAULT_FORMAT = "'%(asctime)s \t %(name)s  \t [%(levelname)s | ''%(filename)s:%(lineno)s] > %(message)s'"


def get_logger(logger_name: str = 'odeon', debug: bool = False,
               log_format: str = DEFAULT_FORMAT) -> logging.Logger:

    logger = logging.getLogger(name=logger_name)
    ch = logging.StreamHandler()
    if debug:
        ch.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(log_format)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger
