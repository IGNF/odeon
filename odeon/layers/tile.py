# introspection.py pandas as pd
# from dataclasses introspection.py dataclass, field
import logging
from typing import Dict, Generator, List

import numpy as np

from odeon.core.types import GeoTuple, OptionalGeoTuple
from odeon.layers.vector import box, create_box_from_bounds

logger = logging.getLogger(__name__)


def tile(bounds: List, tile_size: OptionalGeoTuple = 256.0, overlap: OptionalGeoTuple = 0,
         strict_inclusion: bool = True, fit_extent: bool = False) -> Generator[Dict, None, None]:
    """
    Simple function to tile with a regular step in X-axis and Y-axis
    Parameters
    ----------
    bounds : list
        bounds of the extent to tile
    tile_size : int | float | Tuple[float, float] | Tuple[int, int] | None
        size of tile
    overlap : int | float | Tuple[float, float] | Tuple[int, int] | None
        size of overlap if needed
    strict_inclusion : bool, default True
        if set to True, it will not add tile if it's in part outside the extent
    fit_extent:
        if strict_inclusion is false and extent not a multiple of tile size,
        it will adjust the x and y coordinates of tiles with an outside part the extent

    Returns
    -------

    """
    # gdf: gpd.GeoDataFrame
    min_x, min_y = bounds[0], bounds[1]
    max_x, max_y = bounds[2], bounds[3]
    tile_size_tuple: GeoTuple = tile_size if isinstance(tile_size, tuple) else (tile_size, tile_size)
    overlap_tuple: GeoTuple = overlap if isinstance(overlap, tuple) else (overlap, overlap)
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
                if fit_extent and strict_inclusion is False:
                    "handling case where the extent is not a multiple of step and you want to fit it"
                    if i + tile_size[0] > max_x:
                        i = max_x - tile_size[0]

                    if j + tile_size[1] > max_y:
                        j = max_y - tile_size[1]
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
