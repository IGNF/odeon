# import pandas as pd
# from dataclasses import dataclass, field
import logging
from typing import Dict, Generator, List, Tuple

import geopandas as gpd
import numpy as np

from .types import GeoTuple, OptionalGeoTuple
from .vector import box, create_box_from_bounds

logger = logging.getLogger(__name__)


def tile(bounds: List, tile_size: OptionalGeoTuple = 256.0, overlap: OptionalGeoTuple = 0,
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
    tile_size_tuple: GeoTuple = tuple(tile_size) if isinstance(tile_size, Tuple) else tuple((tile_size, tile_size))
    overlap_tuple: GeoTuple = overlap if isinstance(overlap, Tuple) else (overlap, overlap)
    assert 2 * overlap_tuple[0] < tile_size_tuple[0]
    assert 2 * overlap_tuple[1] < tile_size_tuple[1]

    step = [
        tile_size_tuple[0] - (2 * overlap_tuple[0]),
        tile_size_tuple[1] - (2 * overlap_tuple[1])
    ]

    for i in np.arange(min_x - overlap_tuple[0], max_x + overlap_tuple[0], step[0]):

        for j in np.arange(min_y - overlap_tuple[1], max_y + overlap_tuple[1], step[1]):

            "handling case where the extent is not a multiple of step"
            if (i + tile_size_tuple[0] > max_x + overlap_tuple[0] or j + tile_size_tuple[1] > max_y + overlap_tuple[1])\
                    and strict_inclusion:
                pass
            else:
                # j = max_y + tile_size[1] - tile_size[1]
                left = i
                right = i + tile_size_tuple[0]
                bottom = j
                top = j + tile_size_tuple[1]
                bbox: box = create_box_from_bounds(left, bottom, right, top)
                row_d = {
                    "id": f"{left}-{bottom}-{right}-{top}",
                    "geometry": bbox
                }
                yield row_d
