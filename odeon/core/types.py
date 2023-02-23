from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Literal, Mapping,
                    Optional, Tuple, TypeAlias, Union)

import geopandas as gpd
import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.loggers.logger import Logger
from torchmetrics import Metric, MetricCollection

from .app_utils import Stages

URI: TypeAlias = Union[str, Path]
OPT_URI: TypeAlias = Optional[URI]
URIS: TypeAlias = List[URI]
URI_OR_URIS: TypeAlias = Union[URI, URIS]
OPT_URI_OR_URIS: TypeAlias = Optional[Union[URI, URIS]]
DATASET: TypeAlias = Union[Iterable, Mapping]
PREPROCESS_OPS: TypeAlias = Callable[[Dict], Dict]
STAGES: TypeAlias = Literal[Stages.FIT, Stages.VALIDATE, Stages.TEST, Stages.PREDICT]
STAGES_OR_ALL: TypeAlias = Literal[Stages.FIT, Stages.VALIDATE, Stages.TEST, Stages.PREDICT, 'all']
STAGES_OR_ALL_OR_VALUE: TypeAlias = Literal[Stages.FIT, Stages.VALIDATE, Stages.TEST, Stages.PREDICT, 'all',
                                            'fit', 'validate', 'test', 'predict']
STAGES_OR_VALUE = Literal[Stages.FIT, Stages.VALIDATE, Stages.TEST, Stages.PREDICT,
                          'fit', 'validate', 'test', 'predict']
DATAFRAME: TypeAlias = Union[pd.DataFrame, gpd.GeoDataFrame]
OptionalGeoTuple: TypeAlias = Union[int, float, Tuple[float, float], Tuple[int, int]]
GeoTuple: TypeAlias = Union[Tuple[float, float], List[float]]  # used for stuf like patch size, overlapd, etc.
NoneType: TypeAlias = type(None)
OdnMetric: TypeAlias = Union[Metric, MetricCollection]
OdnLogger: TypeAlias = Logger
OdnCallback: TypeAlias = Callback
PARAMS: TypeAlias = Dict[str, Any]
OdnData: TypeAlias = LightningDataModule
