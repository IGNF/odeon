from enum import Enum

import numpy as np
import rasterio as rio


class InputFormat(Enum):

    FOLDERS = "folders"
    CSV = "csv"
    VECTORFILE = "vector_file"


class InputDataKeys(Enum):

    INPUT = "input"
    PREDS = "preds"
    TARGET = "target"
    METADATA = "metadata"


class InputDType(Enum):
    UINT8 = ["uint8", np.uint8, rio.uint8]
    UINT16 = ["float16", np.uint16, rio.uint16]
    FLOAT32 = ["float32", np.float32, rio.float32]
    FLOAT64 = ["float64", np.float64, rio.float64]


class DTypeRange(Enum):
    UINT8 = (0, 255)
    UINT16 = (0, 65535)
    FLOAT32 = (0, 3.4028235e+38)
    FLOAT64 = (0, 1.7976931348623157e+308)


class TransformStrategy(Enum):
    SAMPLE_WISE = "sample_wise"
    BATCH_WISE = "batch_wise"


class TargetTYPES(Enum):
    MASK = "mask"


RASTER_ACCEPTED_EXTENSION = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".jp2", ".vrt"]
