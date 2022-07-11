from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from .types import DATAFRAME, URI_OR_URIS

CSV_SUFFIX = ".csv"


def create_pandas_dataframe_from_file(path: URI_OR_URIS, options: Optional[Dict] = None) -> pd.DataFrame:
    """

    Parameters
    ----------
    path: Path or str, path of csv file
    options: Dict or None, pandas options
    Returns
    -------
     DataFrame
    """
    return pd.read_csv(path, **options) if options is not None else pd.read_csv(path)


def create_geopandas_dataframe_from_file(path: URI_OR_URIS, options: Optional[Dict] = None) -> gpd.GeoDataFrame:
    """

    Parameters
    ----------
    path : Path or str, path of vector file
    options: Dict or None, geopandas options
    Returns
    -------
     GeoDataFrame
    """
    return gpd.read_file(path, **options) if options is not None else gpd.read_file(path)


def split_dataframe(data: DATAFRAME, split_ratio: float = 0.8) -> Tuple[DATAFRAME, DATAFRAME]:
    """

    Parameters
    ----------
    data
    split_ratio

    Returns
    -------

    """
    msk = np.random.rand(len(data)) < split_ratio
    split_1: DATAFRAME = data[msk]
    split_2: DATAFRAME = data[~msk]
    return split_1, split_2
