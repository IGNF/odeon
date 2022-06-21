from pathlib import Path
import logging
from typing import List, Optional, Any, Union, Tuple, Protocol, Dict, runtime_checkable
from dataclasses import dataclass, field
import geopandas as gpd
from ..core.tile import tile, Overlap, TileSize, build_quad_tree, QuadTree, VALID_CRITERION
from ..core.shape import load_polygon_from_wkt, print_gdf

logger = logging.getLogger(__name__)


@runtime_checkable
class TilerInterface(Protocol):

    def tile(self, options: Optional[Dict[Any, Any]]) -> gpd.GeoDataFrame:
        ...

    def print(self, filename: Union[str, Path], driver: Optional[str]):
        ...


@dataclass
class SimpleTiler:
    """"A dataclass.

    Parameters
    ----------
    """

    tile_size: Union[int, float, List[float], Tuple[float, float]]
    extent: Union[Optional[str], gpd.GeoDataFrame] = field(default=None)
    bounds: Union[Optional[str], List] = field(default=None)
    crs: Optional[Any] = field(default=None)
    debug: bool = False
    overlap: Union[Optional[int], float, List[float], Tuple[float, float]] = 0
    has_extent: bool = field(default=None)
    tiled_gdf: gpd.GeoDataFrame = field(default=None)
    predicate: str = "intersects"
    strict_inclusion: bool = True

    def __post_init__(self):

        assert self.predicate in ["intersects", "within"]
        self._tile_size: TileSize = SimpleTiler.init_type(self.tile_size)
        if self.extent is not None:
            self.extent: gpd.GeoDataFrame = self.extent if isinstance(self.extent,
                                                                      gpd.GeoDataFrame) else gpd.read_file(self.extent)
            self.has_extent = True
            self.bounds = self.extent.geometry.unary_union.bounds if self.bounds is None else self.bounds
            self.crs = self.extent.crs
        self._bounds: List = load_polygon_from_wkt(self.bounds).bounds if isinstance(self.bounds, str) else self.bounds

        if self.crs is None:
            raise AttributeError("crs variable has not been initialized, you can do it"
                                 "by initializing crs or by initializing the extent attribute")
        self._overlap: Overlap = [0.0, 0.0] if self.overlap is None else SimpleTiler.init_type(self.overlap)

    def tile(self) -> gpd.GeoDataFrame:
        """

        Returns
        -------
        gpd.GeoDataFrame

        """

        tile_generator = tile(bounds=self._bounds,
                              tile_size=self._tile_size,
                              overlap=self._overlap,
                              strict_inclusion=self.strict_inclusion)
        tmp_list: List[Dict] = [i for i in tile_generator]
        self.tiled_gdf = gpd.GeoDataFrame(tmp_list, crs=self.crs, geometry="geometry")

        if self.has_extent:
            convex_hull = gpd.GeoDataFrame([{"id_box": 0, "geometry": self.extent.geometry.unary_union.convex_hull}],
                                           crs=self.crs)
            joined = gpd.sjoin(self.tiled_gdf, convex_hull, how="inner", predicate=self.predicate, rsuffix="r")
            self.tiled_gdf = self.tiled_gdf[self.tiled_gdf["id"].isin(joined["id"].unique())]
            return self.tiled_gdf
        else:
            return self.tiled_gdf

    def print(self, filename: Union[str, Path], driver: Optional[str] = None):
        print_gdf(self.tiled_gdf, str(filename), driver)

    @staticmethod
    def init_type(attr: Union[int, float, List[float], Tuple[float, float]]) -> List[float]:

        if isinstance(attr, int):
            return [float(attr), float(attr)]
        elif isinstance(attr, float):
            return [float(attr), float(attr)]
        else:
            return list(attr)


@dataclass
class QuadTreeTiler:
    """"A dataclass.

    Parameters
    ----------
    """

    criterion_value: Union[int, float]
    criterion: str = VALID_CRITERION[0]
    extent: Union[Optional[str], gpd.GeoDataFrame] = field(default=None)
    bounds: Union[Optional[str], List] = field(default=None)
    crs: Optional[Any] = field(default=None)
    debug: bool = False
    has_extent: bool = field(default=None)
    quad_tree: QuadTree = field(default=None)
    predicate: str = "intersects"

    def __post_init__(self):
        assert self.predicate in ["intersects", "within"]
        assert self.criterion in VALID_CRITERION
        if self.criterion == VALID_CRITERION[0] and isinstance(self.criterion_value, float):
            raise AttributeError(f"an integer value is expected  for criterion_value "
                                 f"when you pick the criterion {VALID_CRITERION[0]}")
        self.criterion_value = self.criterion_value if self.criterion == VALID_CRITERION[0] \
            else float(self.criterion_value)
        if self.extent is not None:
            self.extent: gpd.GeoDataFrame = self.extent if isinstance(self.extent,
                                                                      gpd.GeoDataFrame) else gpd.read_file(self.extent)
            self.has_extent = True
            self.bounds = self.extent.geometry.unary_union.bounds if self.bounds is None else self.bounds
            self.crs = self.extent.crs
        self._bounds: List = load_polygon_from_wkt(self.bounds).bounds if isinstance(self.bounds, str) else self.bounds

        if self.crs is None:
            raise AttributeError("crs variable has not been initialized, you can do it"
                                 "by initializing crs or by initializing the extent attribute")

    def tile(self) -> QuadTree:
        self.quad_tree = build_quad_tree(bounds=self._bounds,
                                         criterion=self.criterion,
                                         criterion_value=self.criterion_value,
                                         crs=self.crs)
        logger.info(f"levels of quad tree {self.quad_tree.levels.keys()}")
        if self.has_extent:
            convex_hull = gpd.GeoDataFrame(
                [{"id_box": 0, "geometry": self.extent.geometry.unary_union.convex_hull}], crs=self.crs)
            for k, v in self.quad_tree.levels.items():
                joined = gpd.sjoin(v, convex_hull, how="inner", predicate=self.predicate, rsuffix="r")
                # logger.info(joined.columns)
                logger.info(v.crs)
                logger.info(f"length of joined: {len(joined)}")
                logger.info(f"length of v for level {k}: {len(v)}")
                self.quad_tree.levels[k] = v[v["id"].isin(joined["id"].unique())]
            return self.quad_tree
        else:
            return self.quad_tree

    def print(self, filename: Union[str, Path], driver: Optional[str] = None):
        filename = str(filename)
        sp = filename.split(".")
        for k in self.quad_tree.levels.keys():
            f = f"{sp[0]}-{k}.{sp[1]}"
            print_gdf(self.quad_tree.levels[k], f, driver)
