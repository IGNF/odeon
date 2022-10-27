from enum import Enum
from typing import List


class InputFormat(Enum):

    FOLDERS = "folders"
    CSV = "csv"
    VECTORFILE = "vector_file"


class InputDataKeys(Enum):
    INPUT = "input"
    PREDS = "preds"
    TARGET = "target"
    METADATA = "metadata"


class InputType(Enum):
    RASTER = "raster"


class InputDType(Enum):
    UINT8 = "uint8"
    UINT16 = "uint16"


DTYPE_MAX = {InputDType.UINT8.value: 255.0, InputDType.UINT16.value: 65535.0}


"""
class TransformStrategy(Enum):
    SAMPLE_WISE = "sample_wise"
    BATCH_WISE = "batch_wise"
"""


class TargetTYPES(Enum):
    MASK = "mask"


IMAGE_MODALITY = [InputType.RASTER.value]

RASTER_ACCEPTED_EXTENSION: List[str] = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".jp2", ".vrt"]
