# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import List, Optional

from torch.utils.data import Dataset

from .types import DATAFRAME, PREPROCESS_OPS


class TaskDataset(Dataset):

    def __init__(self,
                 data: DATAFRAME,
                 input_fields: List,
                 preprocess: Optional[PREPROCESS_OPS],
                 transform: Optional[PREPROCESS_OPS]
                 ):
        """
        Parameters
        ----------
        data: DATAFRAME, can be a pandas or CSV dataframe
        input_fields
        preprocess
        transform
        """
        self.data = data
        self.input_fields = input_fields
        self.preprocess = preprocess
        self.transform = transform

    def sanitize_dataframe(self):
        cols = self.data.columns
        for field in self.input_fields:
            assert field in cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        values = self.data.loc[index, [self.input_fields]]
        fields = {field: value for (field, value) in zip(self.input_fields, values)}
        out = self.preprocess(fields)
        out = self.transform(out) if self.transform is not None else out
        return out
