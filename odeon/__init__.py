"""
pmps package
"""
import pathlib
from .core.logger import get_new_logger, get_stream_handler

this_dir = pathlib.Path(__file__).resolve().parent

with (this_dir / ".." / "VERSION").open() as vf:
    __version__ = vf.read().strip()

try:
    LOGGER = get_new_logger(__name__)
    LOGGER.addHandler(get_stream_handler())
except Exception as e:
    print(e)
    raise e
