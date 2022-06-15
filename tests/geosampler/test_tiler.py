"""Test GeoSampler.tile Api
Test configurations
Test that result is as expected
"""
import os
from pathlib import Path
import logging
from typing import List, Union
import pytest
import geopandas as gpd
from odeon.geosampler.api.tile import SimpleTiler, QuadTreeTiler
from odeon.geosampler.core.tile import QuadTree, build_quad_tree, VALID_CRITERION


logger = logging.getLogger(__name__)
# path_tmp = Path(pytest.tmp_dir) / 'tile_test'
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
    (64, t_bounds, test_crs, None, 15, False, "intersects", 'test1_building.geojson', "GeoJSON"),
    (32, None, None, str(path_to_data), 0, True, "intersects", 'test2_building.geojson', "GeoJSON"),
    (128, None, test_crs, geodf, 25, True, "within", 'test3_building.geojson', "GeoJSON")
]


@pytest.mark.parametrize("tile_size, bounds, crs, extent, overlap, strict_inclusion, predicate, output_file, driver",
                         test_simple_tiler_data)
def test_simple_tiler(tile_size: int, bounds: List, crs: str,
                      extent: Union[gpd.GeoDataFrame, str, Path], overlap: int, strict_inclusion: bool,
                      predicate: str, output_file: Union[str, Path], driver: str) -> None:

    simple_tiler = SimpleTiler(tile_size=tile_size, bounds=bounds, crs=crs, extent=extent,
                               overlap=overlap, strict_inclusion=strict_inclusion, predicate=predicate)
    simple_tiler.tile()
    output_file = pytest.tmp_dir / output_file
    simple_tiler.print(output_file, driver=driver)


# bounds: List, criterion: str, criterion_value: Union[int, float], crs: str
test_bounds = [506880.00000000914, 6286848.0, 508416.0, 6288379.599999994]
test_bounds_2 = [500000.0, 6200000.0, 501000.0, 6201000.0]
test_quad_tiler_data = [(test_bounds, VALID_CRITERION[0], 50, test_crs),
                        (test_bounds_2, VALID_CRITERION[1], 2500, test_crs)]


@pytest.mark.parametrize("bounds, criterion, criterion_value, crs",
                         test_quad_tiler_data)
def test_quad_tree_tile(bounds, criterion, criterion_value, crs) -> None:

    quad_tree = build_quad_tree(bounds=bounds,
                                criterion=criterion,
                                criterion_value=criterion_value,
                                crs=crs)
    """
    for k, v in quad_tree.levels.items():

        f = f"{criterion}-level-{k}.geojson"
        v.to_file(f, driver="GeoJSON")
    """
    assert isinstance(quad_tree, QuadTree)


test_quad_tree_tiler = [
                        (t_bounds, VALID_CRITERION[0], 50, test_crs, None,
                         "intersects", 'test1.geojson', "GeoJSON"),
                        (None, VALID_CRITERION[1], 2500, None, str(path_to_data),
                         "intersects", 'test2.geojson', "GeoJSON")
]


@pytest.mark.parametrize("bounds, criterion, criterion_value, crs, extent, predicate, output_file, driver",
                         test_quad_tree_tiler)
def test_quad_tree_tiler(bounds, criterion, criterion_value, crs, extent, predicate, output_file, driver) -> None:
    quad_tree_tiler = QuadTreeTiler(criterion=criterion,
                                    criterion_value=criterion_value,
                                    bounds=bounds,
                                    crs=crs,
                                    extent=extent,
                                    predicate=predicate)
    quad_tree_tiler.tile()
    output_file = pytest.tmp_dir / output_file
    quad_tree_tiler.print(filename=output_file, driver=driver)
    logger.info(pytest.tmp_dir)
    assert isinstance(quad_tree_tiler, QuadTreeTiler)
