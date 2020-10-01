"""Folder manager

This tool is responsible for the creation and destruction of folders, subfolders and files

Notes
-----

"""

import pathlib
import shutil
import os
from os import listdir


def create_folder(path):
    """create folder with the whole hierarchy if required

    Parameters
    ----------
    path complete path of the folder

    Returns
    -------

    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def find_file_names(path_to_dir, suffix=".csv"):
    """

    Parameters
    ----------
    path_to_dir str
    suffix str

    Returns
    -------
    list[str]: list of files
    """

    file_names = listdir(path_to_dir)
    return [os.path.join(path_to_dir, filename) for filename in file_names if filename.endswith(suffix)]


def build_directories(paths, append=True, exist_ok=True):
    """
    Make directory
    Parameters
    ----------
    paths str

    Returns
    -------

    """

    for k, path in paths.items():

        if os.path.isdir(path) and bool(append) is False:

            shutil.rmtree(path)

        os.makedirs(path, exist_ok=exist_ok)
