"""
Module to define custom exception
"""
import traceback
from enum import Enum, auto, unique


class OdeonError(Exception):
    """
    Custom Exception for Odeon project
    """

    def __init__(self, error_code, message='', stack_trace=None, *args, **kwargs):
        """

        Parameters
        ----------
        error_code : ErrorCodes
         code of the error
        message : str
        stack_trace : object
         trace of python Exception possibly with this error

        args
        kwargs
        """

        # Raise a separate exception in case the error code passed isn't specified in the ErrorCodes enum
        if not isinstance(error_code, ErrorCodes):

            msg = 'Error code passed in the error_code param must be of type {0}'
            raise OdeonError(ErrorCodes.ERR_INCORRECT_ERRCODE, msg, args=[ErrorCodes.__class__.__name__])

        # Storing the error code on the exception object
        self.error_code = error_code

        # storing the traceback which provides useful information about where the exception occurred
        self.traceback = traceback.format_exc()
        self.stack_trace = stack_trace if stack_trace is not None else ""

        # Prefixing the error code to the exception message
        try:

            msg = f"{str(message)} \n error code : {str(self.error_code)} \n " \
                  f"trace back: {str(self.traceback)} \n" \
                  f" stack trace: {str(self.stack_trace)} \n" \
                  f" {str(args)} \n str{str(kwargs)}"

        except (IndexError, KeyError):

            msg = f"{error_code.name},  {message}"

        super().__init__(msg)


@unique
class ErrorCodes(Enum):
    """Error codes for all module exceptions

    """

    ER_DEFAULT = auto()

    """  error code passed is not specified in enum ErrorCodes """
    ERR_INCORRECT_ERRCODE = auto()

    """ happens if a raster or a vector is not
     geo referenced """
    ERR_COORDINATE_REFERENCE_SYSTEM = auto()

    """ happens if a raster or a vector has a driver incompatibility with Odeon"""
    ERR_DRIVER_COMPATIBILITY = auto()

    """ happens if we ask or try to access to a non existent band of a raster with Odeon"""
    ERR_RASTER_BAND_NOT_EXIST = auto()

    """ happens if a file (in a json configuration CLI mostly) doesn't exist """
    ERR_FILE_NOT_EXIST = auto()

    """ happens if a dir (in a json configuration CLI mostly) doesn't exist """
    ERR_DIR_NOT_EXIST = auto()

    """ happens when the opening of a file raises an IO error """
    ERR_IO = auto()

    """ happens when a json schema validation raises an error"""
    ERR_JSON_SCHEMA_ERROR = auto()

    """ happens when something goes wrong in generation """
    ERR_GENERATION_ERROR = auto()

    """ happens when something goes wrong in sampling """
    ERR_SAMPLING_ERROR = auto()

    """ happens when something goes wrong in configuration """
    ERR_CONF_ERROR = auto()

    """ happens when a field is not found in any type of key value pair object """
    ERR_FIELD_NOT_FOUND = auto()

    """ happens when a critical test of interection returns false"""
    ERR_INTERSECTION_ERROR = auto()

    """ happens when an iterable object must be not empty"""
    ERR_EMPTY_ITERABLE_OBJECT = auto()

    """happens when an index is out of the bound of an object"""
    ERR_OUT_OF_BOUND = auto()

    """ happens when a path of datset is not valid"""
    INVALID_DATASET_PATH = auto()

    """ happens when something went wrong during the detection """
    ERR_DETECTION_ERROR = auto()

    """ happens when something went wrong during training """
    ERR_TRAINING_ERROR = auto()

    """ happens when we try to build a pytorch model"""
    ERR_MODEL_ERROR = auto()

    def __str__(self):
        """

        Returns
        -------
        str, str
         name of enum member, value of enum member
        """
        return f"name of error: {self.name}, code value of error: {self.value}"


class MisconfigurationException(OdeonError):
    """Exception used to inform users of misuse with Configuration like command line interface variable environment"""

    def __init__(self, error_code=ErrorCodes.ERR_CONF_ERROR, message='', stack_trace=None, *args, **kwargs):
        super(MisconfigurationException, self).__init__(error_code=error_code,
                                                        message=message,
                                                        stack_trace=stack_trace,
                                                        *args,
                                                        **kwargs)
