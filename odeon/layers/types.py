from typing import List, Tuple, Union

import geopandas as gpd
import pandas as pd

DATAFRAME = Union[pd.DataFrame, gpd.GeoDataFrame]
BOUNDS: Union[Tuple[float, float, float, float], Tuple[int, int, int, int], List[int], List[float]]
