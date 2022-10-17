from pathlib import Path
from typing import (Callable, Dict, Iterable, List, Literal, Mapping, Optional,
                    Tuple, Union)

import geopandas as gpd
import pandas as pd

from .runner_utils import Stages

URI = Union[str, Path]
URIS = List[URI]
URI_OR_URIS = Union[URI, URIS]
OPT_URI_OR_URIS = Optional[Union[URI, URIS]]
DATASET = Union[Iterable, Mapping]
PREPROCESS_OPS = Callable[[Dict], Dict]
STAGES = Literal[Stages.FIT, Stages.VALIDATE, Stages.TEST, Stages.PREDICT]
DATAFRAME = Union[pd.DataFrame, gpd.GeoDataFrame]
OptionalGeoTuple = Union[int, float, Tuple[float, float], Tuple[int, int]]
GeoTuple = Union[Tuple[float, float], Tuple[int, int]]  # used for stuf like patch size, overlapd, etc.
