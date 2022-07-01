from typing import Union, Iterable, Mapping, List, Optional, Callable, Dict, Literal
from pathlib import Path
from .runner_utils import Stages
import pandas as pd
import geopandas as gpd
URI = Union[str, Path]
URIS = List[URI]
URI_OR_URIS = Union[URI, URIS]
OPT_URI_OR_URIS = Optional[Union[URI, URIS]]
DATASET = Union[Iterable, Mapping]
SAMPLEWISE_OPS = Callable[[Dict], Dict]
STAGES = Literal[Stages.FIT, Stages.VALIDATE, Stages.TEST, Stages.PREDICT]
DATAFRAME = Union[pd.DataFrame, gpd.GeoDataFrame]
