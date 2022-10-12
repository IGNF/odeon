"""
pmps package
"""
import pathlib

this_dir = pathlib.Path(__file__).resolve().parent

with (this_dir / ".." / "VERSION").open() as vf:
    __version__ = vf.read().strip()
