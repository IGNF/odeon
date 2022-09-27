# import pandas as pd
# from dataclasses import dataclass, field
import logging
from typing import Dict, Generator, List, Union

import geopandas as gpd
import numpy as np

from .vector import box, create_box_from_bounds

Overlap = Union[float, List[float]]
TileSize = Union[int, List]
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
