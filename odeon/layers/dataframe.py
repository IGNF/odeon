from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from odeon.core.types import URI

from .core.types import DATAFRAME

CSV_SUFFIX = ".csv"


def create_dataframe_from_file(path: URI, options: Optional[Dict] = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    path: Path or str, path of csv file
    options: Dict or None, pandas options
    Returns
    -------
     DataFrame
    """
    print(f'path: {path}')
    if str(path).endswith(CSV_SUFFIX):
        return create_pandas_dataframe_from_file(path=path, options=options)
    else:
        return create_geopandas_dataframe_from_file(path=path, options=options)


def create_pandas_dataframe_from_file(path: URI, options: Optional[Dict] = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    path: Path or str, path of csv file
    options: Dict or None, pandas options
    Returns
    -------
     DataFrame
    """

    if options is not None:
        if 'header' not in options.keys():
            options['header'] = "infer"
        if 'header_list' not in options.keys():
            options['header_list'] = None

        header_list = options['header_list']
        len_header_list = len(header_list) if isinstance(header_list, list) else 0
        header = None if options['header'] is False else options['header']
        del options['header']
        del options['header_list']
        df: pd.DataFrame = pd.read_csv(path, header=header, **options)
        if isinstance(header_list, list):
            error_message = f"""header list must have same length than csv columns,
                                                but header has {len_header_list} and columns equals to
                                                 {len(df.columns)}"""
            assert len(header_list) == len(df.columns), error_message
            df.columns = header_list
        return df
    else:
        return pd.read_csv(path)


def create_geopandas_dataframe_from_file(path: URI, options: Optional[Dict] = None) -> gpd.GeoDataFrame:
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
