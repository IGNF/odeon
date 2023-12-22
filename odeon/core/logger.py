"""Logging helper module"""
import logging
from IPython.display import display, HTML
from odeon.core.python_env import RUNNING_IN_JUPYTER

DEFAULT_FORMAT = "'%(asctime)s \t %(name)s  \t [%(levelname)s | ''%(filename)s:%(lineno)s] > %(message)s'"

# ANSI color codes
ANSI_COLORS = {
    'DEBUG': '\033[94m',    # Blue
    'INFO': '\033[92m',     # Green
    'WARNING': '\033[93m',  # Yellow
    'ERROR': '\033[91m',    # Red
    'CRITICAL': '\033[95m', # Magenta
    'ENDC': '\033[0m',      # Reset to default color
}


class JupyterColorFormatter(logging.Formatter):
    def format(self, record):
        color_map = {
            'DEBUG': 'blue',
            'INFO': 'green',
            'WARNING': 'orange',
            'ERROR': 'red',
            'CRITICAL': 'purple'
        }
        color = color_map.get(record.levelname, 'black')
        return f"<span style='color: {color}'>{record.levelname}: {record.getMessage()}</span>"


class ANSIColorFormatter(logging.Formatter):
    def format(self, record):
        color = ANSI_COLORS.get(record.levelname, '')
        endc = ANSI_COLORS['ENDC']
        return f"{color}{record.levelname}: {record.getMessage()}{endc}"


class JupyterLogHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            display(HTML(msg))
        except Exception:
            self.handleError(record)


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
        formatter = JupyterColorFormatter()
    else:
        ch = logging.StreamHandler()
        formatter = ANSIColorFormatter(log_format)

    ch.setFormatter(formatter)
    ch.setLevel(level=level)
    logger.addHandler(ch)
    return logger
