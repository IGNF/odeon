"""
pmps package
"""
import pathlib
from pathlib import Path

HOME = str(Path.home())
DEFAULT_ODEON_PATH = HOME
this_dir = pathlib.Path(__file__).resolve().parent

with (this_dir / ".." / "VERSION").open() as vf:
    __version__ = vf.read().strip()
