"""Test GeoSampler.sample Api
Test configurations
Test that result is as expected
"""
import os
import random
from pathlib import Path
import logging
from typing import List, Union
import pytest
import geopandas as gpd
from odeon.geosampler import GridSampling, RandomSampling
from odeon.geosampler.api.tile import SimpleTiler

logger = logging.getLogger(__name__)
path_to_data: Path = Path(os.path.join(os.path.dirname(__file__), '../../data/example_building.shp'))
geodf = gpd.read_file(path_to_data)
# logger.info(f"bounds: {geodf.geometry.unary_union.bounds}")
t_bounds = [506880.00000000914, 6286848.0, 508416.0, 6288379.599999994]
test_crs = "epsg:2154"

"""
Params to test
tile_size, bounds, extent, overlap, strict_inclusion, ops, output_file
"""
test_simple_tiler_data = [
    (64, t_bounds, test_crs, None, 15, False, "intersects", True, False),
    (32, None, None, str(path_to_data), 0, True, "intersects", False, True),
    (128, None, test_crs, geodf, 25, True, "within", True, False)
]


@pytest.mark.parametrize("tile_size, bounds, crs, extent, overlap, strict_inclusion, predicate, tile, with_box",
                         test_simple_tiler_data)
def test_grid_sampling(tile_size: int, bounds: List, crs: str,
                       extent: Union[gpd.GeoDataFrame, str, Path], overlap: int, strict_inclusion: bool,
                       predicate: str, tile: bool, with_box: bool) -> None:

    simple_tiler = SimpleTiler(tile_size=tile_size, bounds=bounds, crs=crs, extent=extent,
                               overlap=overlap, strict_inclusion=strict_inclusion, predicate=predicate)
    if tile:
        simple_tiler.tile()
    grid_sampler = GridSampling(tiler=simple_tiler, with_box=with_box)
    output = grid_sampler.sample()
    if ~tile:
        simple_tiler.tile()
    if with_box:
        assert len(output) == 2
        assert len(output[0]) == len(simple_tiler.tiled_gdf)
        assert len(output[1]) == len(simple_tiler.tiled_gdf)
    else:
        assert isinstance(output, gpd.GeoDataFrame)
        assert len(output) == len(simple_tiler.tiled_gdf)


@pytest.mark.parametrize("tile_size, bounds, crs, extent, overlap, strict_inclusion, predicate, tile, with_box",
                         test_simple_tiler_data)
def test_random_sampling(tile_size: int, bounds: List, crs: str,
                         extent: Union[gpd.GeoDataFrame, str, Path], overlap: int, strict_inclusion: bool,
                         predicate: str, tile: bool, with_box: bool) -> None:
    simple_tiler = SimpleTiler(tile_size=tile_size, bounds=bounds, crs=crs, extent=extent,
                               overlap=overlap, strict_inclusion=strict_inclusion, predicate=predicate)

    simple_tiler.tile()
    gdf = simple_tiler.tiled_gdf
    n_sample = random.randint(1, len(gdf) - 1)
    random_sampler = RandomSampling(tiler=simple_tiler, with_box=with_box, n_sample=n_sample)
    output = random_sampler.sample()

    if with_box:
        assert len(output) == 2
        assert len(output[0]) == n_sample
        assert len(output[1]) == n_sample
    else:
        assert isinstance(output, gpd.GeoDataFrame)
        assert len(output) == n_sample
