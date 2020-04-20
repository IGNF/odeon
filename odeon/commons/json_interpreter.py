"""Json Interpreter

This class checks the content of json files:
* section and section content
* default values

Notes
-----
    * [Todo] implement default values
    * [Todo] is a section is absent, check if default values exists

"""

import json


class JsonInterpreter:
    """
    Json dictionary with default values (TODO).

    ...

    Attributes
    ----------
    __dict__ : dictionary
        list of (parameters, values).

    Methods
    -------
    get_xxx()
        retrieve the data for a specific section.
    check_content(tag_names)
        check the existence of the section in the dictionary.

    """

    def __init__(self, json_file):
        self.__dict__ = json.load(json_file)

    def get_image(self):
        if "image" in self.__dict__:
            return self.__dict__["image"]

    def get_sampler(self):
        if "sampler" in self.__dict__:
            return self.__dict__["sampler"]

    def check_content(self, tag_names):
        try:
            for name in tag_names:
                if name not in self.__dict__:
                    raise ValueError(f"'{name}' section is missing in json")
        except ValueError as ve:
            print(f"[ERROR] {ve}")
            exit(1)
