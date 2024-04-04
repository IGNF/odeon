"""Logging helper module"""
import logging

from IPython.display import HTML, display

from odeon.core.python_env import RUNNING_IN_JUPYTER

DEFAULT_FORMAT = "'%(asctime)s \t %(name)s  \t [%(levelname)s | ''%(filename)s:%(lineno)s] > %(message)s'"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ANSI color codes
ANSI_COLORS = {
    'DEBUG': '\033[94m',    # Blue
    'INFO': '\033[92m',     # Green
    'WARNING': '\033[93m',  # Yellow
    'ERROR': '\033[91m',    # Red
    'CRITICAL': '\033[95m',  # Magenta
    'ENDC': '\033[0m',      # Reset to default color
}


# Custom log handler for Jupyter Notebook
class JupyterLogHandler(logging.Handler):
    def emit(self, record):
        try:
            message = self.format(record)
            display(HTML(message))
        except Exception:
            self.handleError(record)


class JupyterColorFormatter(logging.Formatter):
    # Color map for HTML
    HTML_COLORS = {
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'orange',
        'ERROR': 'red',
        'CRITICAL': 'purple'
    }

    def format(self, record):
        color = self.HTML_COLORS.get(record.levelname, 'black')
        message = super().format(record)
        return f"<span style='color: {color}'>{message}</span>"


class ANSIColorFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

    def format(self, record):
        levelname = record.levelname
        message = super().format(record)
        if levelname in ANSI_COLORS:
            return ANSI_COLORS[levelname] + message + ANSI_COLORS['ENDC']
        return message


def get_logger(logger_name: str = 'odeon', debug: bool = False,
               log_format: str = DEFAULT_FORMAT) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if debug:
        level = logging.DEBUG
        logger.setLevel(logging.DEBUG)

    else:
        level = logging.INFO
        logger.setLevel(logging.INFO)

    logger.propagate = False

    if RUNNING_IN_JUPYTER:
        ch = JupyterLogHandler()
        formatter = JupyterColorFormatter(fmt=log_format, datefmt=DATE_FORMAT)
    else:
        ch = logging.StreamHandler()
        formatter = ANSIColorFormatter(fmt=log_format, datefmt=DATE_FORMAT)

    ch.setFormatter(formatter)
    ch.setLevel(level=level)
    logger.addHandler(ch)
    return logger
