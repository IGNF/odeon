# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import random
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch.utils.data import Dataset

from odeon.core.types import OptionalGeoTuple
from odeon.data.core.dataloader_utils import (DEFAULT_OVERLAP,
                                              DEFAULT_PATCH_RESOLUTION,
                                              DEFAULT_PATCH_SIZE)
from odeon.data.preprocess import PreProcessor
from odeon.layers.core.types import DATAFRAME
from odeon.layers.tile import tile
from odeon.layers.vector import create_gdf_from_list, gpd

logger = getLogger(__name__)


class OdnDataset(Dataset):
    """
    A dataset class for handling georeferenced data within the PyTorch framework, designed to support
    both training and inference stages. It facilitates operations such as data tiling, preprocessing,
    and applying transformations, with special support for geospatial data zones and patch-based
    processing.

    Parameters
    ----------
    data : DATAFRAME
        The input data, which can be a pandas DataFrame or a geopandas GeoDataFrame for georeferenced data.
    input_fields : Dict
        A dictionary specifying the input fields and their configurations.
    root_dir : Union[None, Path, str], optional
        The root directory from where to load the data files. Defaults to None.
    cache_dataset : bool, optional
        If True, the dataset will be cached, improving performance at the cost of memory. Defaults to False.
    transform : Optional[Callable], optional
        An optional transformation to be applied to the data. This can be a composition of multiple
        transformations. Defaults to None.
    by_zone : bool, optional
        If True, processes the data by geographical zones. This requires the data to be a geopandas
        GeoDataFrame. Defaults to False.
    inference_mode : bool, optional
        Indicates if the dataset is used for inference. In this mode, certain preprocessing steps might
        be skipped or added. Defaults to False.
    patch_size : Tuple[int, int], optional
        The size of the patches to extract from the data, specified as a width and height tuple.
        Defaults to DEFAULT_PATCH_SIZE.
    patch_resolution : Tuple[float, float], optional
        The resolution of the patches to be extracted, specified as a width and height tuple.
        Defaults to DEFAULT_PATCH_RESOLUTION.
    random_window : bool, optional
        If True, windows for patch extraction will be chosen randomly within the bounds of the data,
        providing data augmentation. Defaults to True.
    overlap : OptionalGeoTuple, optional
        The overlap between patches, if any, as a tuple specifying the overlap in the x and y dimensions.
        Defaults to DEFAULT_OVERLAP.
    debug : bool, optional
        If True, enables debug mode, which might print additional information for debugging purposes.
        Defaults to False.

    Methods
    -------
    __len__(self)
        Returns the number of items in the dataset.
    __getitem__(self, index)
        Retrieves the item at the specified index with optional preprocessing and transformation applied.

    Examples
    --------
    >>> from odeon.core.types import DATAFRAME
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> # Example data setup
    >>> data = pd.DataFrame({...})  # Assuming this is a DataFrame with geospatial data
    >>> input_fields = {"field1": "type1", "field2": "type2"}
    >>> dataset = OdnDataset(data=data, input_fields=input_fields, root_dir=Path("/path/to/data"))
    >>> print(len(dataset))
    >>> item = dataset[0]  # Get the first item

    Notes
    -----
    - The `data` parameter can either be a pandas DataFrame or a geopandas GeoDataFrame. When using geospatial
      operations, the latter is required.
    - The class is designed to be flexible, supporting various types of geospatial data processing and analysis tasks,
      including but not limited to, image segmentation, object detection, and classification within georeferenced
       datasets.
    """
    def __init__(self,
                 data: DATAFRAME,
                 input_fields: Dict,
                 root_dir: Union[None, Path, str] = None,
                 cache_dataset: bool = False,
                 transform: Optional[Callable] = None,
                 by_zone: bool = False,
                 inference_mode: bool = False,
                 patch_size: Tuple[int, int] = DEFAULT_PATCH_SIZE,
                 patch_resolution: Tuple[float, float] = DEFAULT_PATCH_RESOLUTION,
                 random_window: bool = True,
                 overlap: OptionalGeoTuple = DEFAULT_OVERLAP,
                 debug: bool = False
                 ):
        """
        Parameters
        ----------
        data: DATAFRAME, can be a pandas or CSV dataframe
        input_fields: Dict
        transform: Callable for applying transformation
        """
        self.data: DATAFRAME = data
        self._zone = copy.deepcopy(self.data)
        self.by_zone = by_zone
        self.inference_mode = inference_mode
        self.patch_size = patch_size
        self.patch_resolution: Tuple[float, float] = patch_resolution \
            if patch_resolution is not None else DEFAULT_PATCH_RESOLUTION
        self.patch_size_u = float(self.patch_size[0] * self.patch_resolution[0]), float(
            self.patch_size[1] * self.patch_resolution[1])
        self.random_window = random_window
        self.overlap = overlap if isinstance(overlap, tuple) else (overlap, overlap)
        self._crs = self.data.crs if isinstance(self.data, gpd.GeoDataFrame) else None
        self._debug = debug
        # case inference by zone, we split
        # TODO this part should be in the DataFactory
        if self.by_zone and self.inference_mode:
            assert isinstance(self.data, gpd.GeoDataFrame)
            d = []
            for idx, i in self.data.iterrows():
                zone_row = i.to_dict()
                del zone_row['geometry']
                # TODO add strict_inclusion option
                for row in tile(bounds=i.geometry.bounds,
                                overlap=self.overlap,
                                tile_size=self.patch_size_u):
                    row.update(zone_row)
                    assert 'geometry' in row.keys()
                    d.append(row)

            self.data = create_gdf_from_list(d, crs=self._crs)
        self.preprocess = PreProcessor(input_fields=input_fields,
                                       by_zone=self.by_zone,
                                       cache_dataset=cache_dataset,
                                       root_dir=root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    @property
    def get_zone(self) -> DATAFRAME:
        return self._zone

    def __getitem__(self, index: int):
        """
        1/ retrieve row for index
        2/ Compute bounds
            Case Training Mode:
                case by zone: compute center window or random window
                case by patch: do nothing
            Case Inference Mode:
                case by zone: use bounds of geometry
                case by patch: do nothing
        3/ Preprocess
        4/ Transform if necessary (transform is not None)
        5/ return data
        Parameters
        ----------
        index

        Returns
        -------

        """
        # 1/
        row = self.data.iloc[index]
        bounds = None
        # 2/
        if self.by_zone:
            bounds = row.geometry.bounds if self.inference_mode else \
                OdnDataset._compute_window(bounds=row.geometry.bounds,
                                           patch_resolution=self.patch_resolution,
                                           random_window=self.random_window,
                                           patch_size=self.patch_size)
        # 3/
        out = self.preprocess(dict(row), bounds=bounds)
        # print(out)
        # print(type(out))
        # 4/
        if self.transform is not None:
            out = self.transform(out)
        # 5/
        return out

    @staticmethod
    def _compute_window(bounds: List,
                        patch_resolution: Tuple[float, float] = DEFAULT_PATCH_RESOLUTION,
                        random_window: bool = True,
                        patch_size: Tuple[int, int] = DEFAULT_PATCH_SIZE) -> List:
        """
        Parameters
        ----------
        bounds
        patch_resolution
        random_window
        patch_size

        Returns
        -------

        """

        patch_size_u: Tuple[float, float] = (patch_size[0] * patch_resolution[0], patch_size[1] * patch_resolution[1])
        if (bounds[2] - bounds[0] <= patch_size_u[0]) or (bounds[3] - bounds[1] <= patch_size_u[1]) \
                or random_window is False:
            # case where patch requested is too big for random crop of window or option to False
            center_x, center_y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2

        else:
            # random crop
            center_x = random.uniform(bounds[0] + patch_size_u[0], bounds[2] - patch_size_u[0])
            center_y = random.uniform(bounds[1] + patch_size_u[1], bounds[3] - patch_size_u[1])

        patch_bounds = [center_x - (patch_size_u[0] / 2),
                        center_y - (patch_size_u[1] / 2),
                        center_x + (patch_size_u[0] / 2),
                        center_y + (patch_size_u[1] / 2)
                        ]
        return patch_bounds
