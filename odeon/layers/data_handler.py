"""Dataclass to handle dataframe and Layer Collection in a pythonic approach"""
from dataclasses import dataclass, field
from typing import Dict, Optional

from geopandas import GeoDataFrame
from pandas import DataFrame

from odeon.core.types import PARAMS, URI

from ._modality import Modality, Sample
from .dataframe import create_dataframe_from_file
from .modality_collection import ModalityCollection
from .types import BOUNDS, DATAFRAME


@dataclass(frozen=False)
class DataHandler:
    """Dataclass to handle dataframe and Layer Collection in a pythonic way"""
    dataframe: DATAFRAME | URI
    modality_params: Dict[str, PARAMS | Modality]
    dataframe_options: PARAMS | None = None
    _geo_referenced_dataframe: bool = field(init=False)
    _modality_collection: Optional[ModalityCollection] = field(init=False)

    def __post_init__(self):
        if not isinstance(self.dataframe, DataFrame):
            self.dataframe = create_dataframe_from_file(path=str(self.dataframe), options=self.dataframe_options)
        self._geo_referenced_dataframe = True if isinstance(self.dataframe, GeoDataFrame) else False
        # TODO instanciate modality collection
        self._modality_collection = ModalityCollection(modality_params=self.modality_params)

    def read(self,
             bounds: Optional[BOUNDS] = None,
             as_dict: bool = False,
             *args,
             **kwargs,) -> Sample | Optional[PARAMS]:
        ...

    def write(self, *args, **kwargs):
        ...
