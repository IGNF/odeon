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
from typing import Callable, Dict, List, Optional, Union

from torch.utils.data import Dataset

from .preprocess import UniversalPreProcessor
from .tile import tile
from .types import DATAFRAME, Overlap
from .vector import create_gdf_from_list

logger = getLogger(__name__)


class UniversalDataset(Dataset):

    def __init__(self,
                 data: DATAFRAME,
                 input_fields: Dict,
                 root_dir: Union[None, Path, str] = None,
                 cache_dataset: bool = False,
                 transform: Optional[Callable] = None,
                 by_zone: bool = False,
                 inference_mode: bool = False,
                 patch_size: int = 256,
                 patch_resolution: List[float] = None,
                 random_window: bool = True,
                 overlap: Overlap = 0.0
                 ):
        """
        Parameters
        ----------
        data: DATAFRAME, can be a pandas or CSV dataframe
        input_fields: Dict
        transform: Callable for applying transformation
        """
        self.data = data
        self._zone = copy.deepcopy(self.data)
        self.by_zone = by_zone
        self.inference_mode = inference_mode
        self.patch_size = patch_size
        self.patch_resolution: List[float] = patch_resolution if patch_resolution is not None else [0.2, 0.2]
        self.random_window = random_window
        self.overlap = overlap
        self._crs = self.data.crs

        # case inference by zone, we split
        if self.by_zone and self.inference_mode:
            self.data = [{"id_zone": idx, "geometry": j} for idx, i in self.data.iterrows()
                         for j in tile(bounds=i.geometry.bounds, overlap=self.overlap)]
            self.data = create_gdf_from_list(self.data, crs=self._crs)
        self.preprocess = UniversalPreProcessor(input_fields=input_fields,
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
            bounds = row.geometry.bounds if self.inference_mode else UniversalDataset._compute_window(
                bounds=row.geometry.bounds,
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
                        patch_resolution: List[float],
                        random_window: bool = True,
                        patch_size: int = 256) -> List:
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

        patch_size_u = [patch_size * patch_resolution[0], patch_size * patch_resolution[1]]
        if (bounds[2] - bounds[0] <= patch_size_u[0]) or (bounds[3] - bounds[1] <= patch_size_u[1])\
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
