"""Folder manager

This tool is responsible for the creation and destruction of folders, subfolders and files

Notes
-----
    * [Todo] implement destruction of folders (for a clean start) or files

"""

import pathlib


def create_folder(path):
    """
    create folder with the whole hierarchy if required
    :param path: complete path of the folder
    :return:
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
