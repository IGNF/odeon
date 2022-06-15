import pandas as pd
from dataclasses import dataclass, field
from typing import List, Union, Generator, Dict
import logging
import numpy as np
import geopandas as gpd
from .shape import create_box_from_bounds, box

Overlap = Union[float, List[float]]
TileSize = Union[int, List]
VALID_CRITERION: List[str] = ["max_sample", "min_sample_area"]
logger = logging.getLogger(__name__)


def tile(bounds: List, tile_size: TileSize = 256, overlap: Overlap = 0,
         strict_inclusion: bool = True) -> Generator[Dict, None, None]:
    """
    Simple function to tile with a regular step in X-axis and Y-axis
    Parameters
    ----------
    bounds : list
        bounds of extent to tile
    tile_size :
    overlap
    strict_inclusion

    Returns
    -------

    """
    gdf: gpd.GeoDataFrame
    min_x, min_y = bounds[0], bounds[1]
    max_x, max_y = bounds[2], bounds[3]
    tile_size = tile_size if isinstance(tile_size, List) else [tile_size, tile_size]
    overlap = overlap if isinstance(overlap, List) else [overlap, overlap]
    assert 2 * overlap[0] < tile_size[0]
    assert 2 * overlap[1] < tile_size[1]

    step = [
        tile_size[0] - (2 * overlap[0]),
        tile_size[1] - (2 * overlap[1])
    ]

    for i in np.arange(min_x - overlap[0], max_x + overlap[0], step[0]):

        for j in np.arange(min_y - overlap[1], max_y + overlap[1], step[1]):

            "handling case where the extent is not a multiple of step"
            if (i + tile_size[0] > max_x + overlap[0] or j + tile_size[1] > max_y + overlap[1]) and strict_inclusion:
                pass
            else:
                # j = max_y + tile_size[1] - tile_size[1]
                left = i
                right = i + tile_size[0]
                bottom = j
                top = j + tile_size[1]
                bbox: box = create_box_from_bounds(left, bottom, right, top)
                row_d = {
                    "id": f"{left}-{bottom}-{right}-{top}",
                    "geometry": bbox
                }
                yield row_d


@dataclass
class QuadTree:

    levels: Dict[int, gpd.GeoDataFrame] = field(default_factory=lambda: dict())

    def get_level(self, index: int) -> gpd.geodataframe:
        """
        Retrieve the respective geoDataFrame of the
        given index
        Parameters
        ----------
        index index number to in the Quadtree

        Returns
        -------

        """
        return self.levels[index]

    def get_last_level(self) -> int:
        """
        Get last inserted level of the Tree
        Returns
        -------
         int
        """
        if len(self.levels) <= 0:
            return 0
        else:
            return max(self.levels.keys())

    def add_level(self, level: gpd.GeoDataFrame) -> int:
        """

        Parameters
        ----------
        level

        Returns
        -------

        """
        new_index: int = self.get_last_level() + 1
        self.levels[new_index] = level
        return new_index


def build_quad_tree(bounds: List, criterion: str, criterion_value: Union[int, float], crs: str) -> QuadTree:
    """

    Produce a recursive tree with balanced node number (4) in a spatial context.

    References
    ----------
    Stevens & Olsen, 2004, Spatially Balanced Sampling of Natural Resources
    .. _web_link: https://cfpub.epa.gov/ncer_abstracts/index.cfm/fuseaction/display.files/fileID/13339

    Parameters
    ----------
    bounds
    criterion
    criterion_value
    crs

    Returns
    -------

    """
    logging.info(f"length bounds {len(bounds)}")
    logging.info(f"bounds {bounds}")
    if criterion not in VALID_CRITERION:
        raise AttributeError(f"you criterion {criterion} is not a valid value, it sould be one of: "
                             f" { str(VALID_CRITERION) }")
    gdf: gpd.GeoDataFrame = gpd.GeoDataFrame([{"id": "",
                                               "geometry": create_box_from_bounds(bounds[0],
                                                                                             bounds[1],
                                                                                             bounds[2],
                                                                                             bounds[3])}],
                                             crs=crs)
    levels = QuadTree({0: gdf})

    def split_dataframe_elements() -> List:
        tmp = []
        for idx, row in gdf.iterrows():

            tmp.extend(split_bound(row))
        return tmp

    def split_bound(row: pd.Series) -> List:
        """
        Split bound in 4 pieces:

        ---------------
        - NE(1) NW(3) -
        - SE(0) SW(2) -
        ---------------

        Parameters
        ----------
        row

        Returns
        -------
        List
        """
        row_bounds = row.geometry.bounds
        min_x, min_y = row_bounds[0], row_bounds[1]
        max_x, max_y = row_bounds[2], row_bounds[3]

        north_east_geo = create_box_from_bounds(min_x, (min_y + max_y) / 2, (min_x + max_x) / 2, max_y)
        south_east_geo = create_box_from_bounds(min_x, min_y, (min_x + max_x) / 2, (min_y + max_y) / 2)
        south_west_geo = create_box_from_bounds((min_x + max_x) / 2, min_y, max_x, (min_y + max_y) / 2)
        north_west_geo = create_box_from_bounds((min_x + max_x) / 2, (min_y + max_y) / 2, max_x, max_y)
        return [{"id": f"{str(row.id)}0", "geometry": south_east_geo},
                {"id": f"{str(row.id)}1", "geometry": north_east_geo},
                {"id": f"{str(row.id)}2", "geometry": south_west_geo},
                {"id": f"{str(row.id)}3", "geometry": north_west_geo}]

    def criterion_reached_at_next_split(n_sample: int, sample_size: Union[int, float]) -> bool:
        """
        Check if criterion has been reached
        Parameters
        ----------
        n_sample
        sample_size

        Returns
        -------
         boolean (True if criterion is reached False otherwise)
        """

        if criterion == VALID_CRITERION[0]:  # max sample case
            if n_sample > criterion_value:
                return True
            else:
                return False
        else:  # min sample area case
            if float(sample_size) < float(criterion_value):
                return True
            else:
                logger.info(f"sample size: {sample_size}")
                return False

    while True:
        n: int = len(gdf)
        # b: List = gdf.iloc[0].geometry.bounds
        s: float = gdf.iloc[0].geometry.area

        if criterion_reached_at_next_split(n_sample=n, sample_size=s):
            # logger.info(f"break with length gdf {len(gdf)} and levels {levels.get_last_level()}")
            break
        else:
            new_level = split_dataframe_elements()
            gdf = gpd.GeoDataFrame(new_level, crs=crs)
            levels.add_level(gdf)
    return levels
