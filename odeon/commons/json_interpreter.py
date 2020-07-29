"""Json Interpreter

This class checks the content of json files:
* section and section content
* default values

Notes
-----


"""

import json
from jsonschema import Draft7Validator, validators, exceptions
from odeon import LOGGER
import sys


class JsonInterpreter:
    """


    ...

    Attributes
    ----------
    __dict__ : dictionary
    _path: string
        list of (parameters, values).

    Methods
    -------
    get_xxx()
        retrieve the data for a specific section.
    check_content(tag_names)
        check the existence of the section in the dictionary.
    get_section(section)
        retrieve specific section of the __init__ dict
    is_valid(json_schema)
        validate __init__ based on a json schema

    """
    @staticmethod
    def extend_with_default(validator_class):
        """

        :param validator_class:
        :return:
        """
        validate_properties = validator_class.VALIDATORS["properties"]

        def set_defaults(validator, properties, instance, schema):
            for prop, sub_schema in properties.items():
                if "default" in sub_schema:
                    instance.setdefault(prop, sub_schema["default"])

            for error in validate_properties(
                    validator, properties, instance, schema,
            ):
                yield error

        return validators.extend(
            validator_class, {"properties": set_defaults},
        )

    def __init__(self, json_file: json):
        """

        :param json_file: string , path to a json file with an image and a sampler section
        """
        self.__dict__ = json.load(json_file)
        self._path = json_file
        # use of last json schema draft, the 7 (https://json-schema.org)
        self.DefaultValidatingDraft7Validator = JsonInterpreter.extend_with_default(Draft7Validator)

    def is_valid(self, json_schema):
        """
        validate the json object in self.__dict__ with proper json schema (see jsonschema library)
        https://json-schema.org/understanding-json-schema/reference/
        :param json_schema: dict with json schema
        :return: boolean
        """
        try:
            self.DefaultValidatingDraft7Validator(json_schema).validate(self.__dict__)
            return True
        except exceptions.ValidationError as ve:
            LOGGER.error("validation error with  you json file {} \n detail: {}".format(self._path, ve))
            return False

    def get_section(self, section):
        """

        :return: dict or array from section section
        """
        if section in self.__dict__:
            return self.__dict__[section]

#######################################################################
#
#   OLD FUNCTIONS; LEGACY
#
#######################################################################

    def get_image(self):
        """

        :return: dict from image section
        """
        if "image" in self.__dict__:
            return self.__dict__["image"]

    def get_sampler(self):
        """

        :return: dict from  sampler section
        """
        if "sampler" in self.__dict__:
            return self.__dict__["sampler"]

    def get_dict(self):
        return self.__dict__

    def check_content(self, tag_names):
        """

        Parameters
        ----------
        tag_names

        Returns
        -------

        """
        try:
            for name in tag_names:
                if name not in self.__dict__:
                    raise ValueError(f"'{name}' section is missing in json")
        except ValueError as ve:
            LOGGER.error(f"[ERROR] {ve}")
            sys.exit(1)
