import random

import logging
from dataclasses import dataclass, field
from typing import Union, Tuple, Protocol, runtime_checkable, Callable, List
import geopandas as gpd
from .tile import TilerInterface, SimpleTiler

logger = logging.getLogger(__name__)


@runtime_checkable
class SamplingInterface(Protocol):

    def sample(self) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]: ...


class TiledSamplingInterface(SamplingInterface, Protocol):

    tiler: TilerInterface


@dataclass
class GridSampling(TiledSamplingInterface, Callable):
    tiler: SimpleTiler
    with_box: bool = False

    def sample(self) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]:

        return self._get_output(gdf=self._get_tiled_gdf())

    def _get_tiled_gdf(self) -> gpd.GeoDataFrame:
        if isinstance(self.tiler.tiled_gdf, gpd.GeoDataFrame):
            return self.tiler.tiled_gdf
        else:
            return self.tiler.tile()

    def _get_output(self, gdf) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]:
        point_gdf = gdf.apply(lambda x: x.geometry.centroid, axis=1)
        point_gdf = gpd.GeoDataFrame(point_gdf, crs=gdf.crs)
        output = (point_gdf, gdf) if self.with_box else point_gdf
        return output

    def __call__(self) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]:
        return self.sample()


@dataclass
class RandomSampling(GridSampling):
    n_sample: int = 50

    def sample(self) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]:

        gdf = self._get_tiled_gdf()
        if self.n_sample > len(gdf):
            logger.warning(f"""
            you have requested a number of sample
            superior to the number of unit of your tiled extent:
            number of sample requested {self.n_sample}, number of tile possible: {len(gdf)}
            We have returned the number of possible tile
            """)
            return self._get_output(gdf=gdf)

        gdf = gdf.sample(n=self.n_sample)
        return self._get_output(gdf=gdf)


@dataclass
class SystematicSampling(RandomSampling):
    """
    from wikipedia:
    systematic sampling is a statistical method
    involving the selection of elements
    from an ordered sampling frame. The most common form
    of systematic sampling is an equiprobability method.
    In this approach, progression through the list is treated circularly,
    with a return to the top once the end of the list is passed.
    The sampling starts by selecting an element from the list at
    random and then every kth element in the frame is selected, where k,
    is the sampling interval (sometimes known as the skip): this is calculated as:[1]
    k = N / n

    Parameters
    ----------
     interval: int default=1
        length of the step used to travel in the study space
     oversampling: bool default=False
     weights: Optional[str] default=None
     random_starting_point: bool default=True

    References
    ----------
    Wikipedia article on systematic sampling
    .. _web_link: https://en.wikipedia.org/wiki/Systematic_sampling

    """

    interval: int = 1
    oversampling: bool = False
    weights: Union[str, List, None] = field(default=None)
    random_starting_point: bool = True
    max_cycle_number: int = 10
    _starting_point: int = field(init=False, repr=False)
    _picked: List = field(default_factory=lambda: list())

    def __post_init__(self):
        self._get_tiled_gdf()
        self._starting_point = random.randint(0, len(self.tiler.tiled_gdf) - 1)
        match self.weights:
            case None:
                self.weights = [1 for i in range(len(self.tiler.tiled_gdf))]
            case str():
                self.weights = self.tiler.tiled_gdf[self.weights]
            case list():
                assert len(self.weights) == len(self.tiler.tiled_gdf)

    def sample(self) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]:
        ...
