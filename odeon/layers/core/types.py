from typing import List, Tuple, TypeAlias, Union

import geopandas as gpd
import pandas as pd

DATAFRAME: TypeAlias = Union[pd.DataFrame, gpd.GeoDataFrame]
BOUNDS: TypeAlias = Union[Tuple[float, float, float, float], Tuple[int, int, int, int], List[int], List[float]]
