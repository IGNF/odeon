# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Callable, Dict, Optional

from torch.utils.data import Dataset

from .preprocess import UniversalPreProcessor
from .types import DATAFRAME


class UniversalDataset(Dataset):

    def __init__(self,
                 data: DATAFRAME,
                 input_fields: Dict,
                 transform: Optional[Callable] = None
                 ):
        """
        Parameters
        ----------
        data: DATAFRAME, can be a pandas or CSV dataframe
        input_fields: Dict
        transform: Callable for applying transformation
        """
        self.data = data
        self.preprocess = UniversalPreProcessor(input_fields=input_fields)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        out = self.preprocess(dict(self.data.iloc[index]))
        if self.transform is not None:
            out = self.transform(out)
        return out
