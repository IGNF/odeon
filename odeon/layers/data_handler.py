"""Dataclass to handle dataframe and Layer Collection in a pythonic approach"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from geopandas import GeoDataFrame
from layers.core.types import BOUNDS, DATAFRAME
from pandas import DataFrame

from odeon.core.types import PARAMS, URI

from .core.modality import Modality, Sample
from .dataframe import create_dataframe_from_file
from .modality_collection import ModalityCollection
from .vector import spatial_join_within_bounds


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False, kw_only=True)
class DataHandler:
    """Dataclass to handle dataframe and Layer Collection in a pythonic way"""
    dataframe: DATAFRAME | URI
    """dictionary where k is the field name of the corresponding column in the dataframe and v
    the params of the modality"""
    modality_params: Dict[str, PARAMS | Modality]
    dataframe_options: PARAMS | None = None
    prefix: URI | None = None
    _geo_referenced_dataframe: bool = field(init=False)
    _modality_collection: ModalityCollection = field(init=False)
    _dataframe: DATAFRAME = field(init=False)

    @property
    def geo_referenced_dataframe(self) -> bool:
        return self._geo_referenced_dataframe

    @geo_referenced_dataframe.setter
    def geo_referenced_dataframe(self, value: bool) -> None:
        self._geo_referenced_dataframe = value

    @property
    def modality_collection(self) -> ModalityCollection:
        return self._modality_collection

    @modality_collection.setter
    def modality_collection(self, value: ModalityCollection) -> None:
        self._modality_collection = value

    def __post_init__(self):
        if not isinstance(self.dataframe, DataFrame):
            self._dataframe = create_dataframe_from_file(path=str(self.dataframe), options=self.dataframe_options)
        else:
            self._dataframe = self.dataframe
        self._geo_referenced_dataframe = True if isinstance(self.dataframe, GeoDataFrame) else False
        self._modality_collection = ModalityCollection(modality_params=self.modality_params,
                                                       geo_referenced=self._geo_referenced_dataframe)

    def get_row(self, idx: int) -> Dict[str, Any]:
        return dict(self._dataframe.iloc[idx])

    def get_rows_from_bounds(self,
                             bounds: BOUNDS,
                             how: str = 'inner',
                             predicate: str = 'intersects') -> list[dict]:
        if self.geo_referenced_dataframe:
            rows = spatial_join_within_bounds(gdf=self._dataframe, bounds=bounds, how=how, predicate=predicate)
            return rows
        else:
            raise NotImplementedError("dataframe is not geo_referenced, you can't get ")

    def read(self,
             idx: int,
             bounds: Optional[BOUNDS] = None,
             as_dict: bool = False,
             **kwargs,) -> Sample | PARAMS:
        # row = self.get_row(idx=idx)
        if self.is_geo_referenced() and self.modality_collection.is_geo_referenced():
            data = self.modality_collection.read(bounds=bounds, **kwargs)
        else:
            data = self.modality_collection.read(**kwargs)
        if as_dict:
            return Sample(data=data)
        else:
            return data

    def write(self,
              idx: int,
              data: Dict[str, Any],
              bounds: Optional[BOUNDS] = None,
              **kwargs):
        # row = self.get_row(idx=idx)
        if self.is_geo_referenced() and self.modality_collection.is_geo_referenced():
            self.modality_collection.write(data=data, bounds=bounds, **kwargs)
        else:
            self.modality_collection.read(data=data, **kwargs)

    def is_geo_referenced(self) -> bool:
        return self.geo_referenced_dataframe
