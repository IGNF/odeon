from typing import Callable, Dict, List, Optional

import albumentations as A
import numpy as np
import torch


class AlbuTransform:

    def __init__(self,
                 input_fields: Dict,
                 pipe: Optional[List[Callable]] = None
                 ):

        self._input_fields = input_fields
        self._pipe: List = list() if pipe is None else pipe
        self._pipe.append(ToTensorCustom())
        self._additional_targets: Dict = dict()
        for key, value in self._input_fields:
            if value["type"] == "raster":
                self._additional_targets[key] = 'image'
            if value["type"] == "mask":
                self._additional_targets[key] = 'mask'
        self.transfrom = A.Compose(self._pipe, additional_targets=self._additional_targets)

    def __call__(self, data: Dict, *args, **kwargs):
        ...
        # transform_data = {key: value for key, value in data.items() if key in self._additional_targets.keys()}


class ToTensorCustom(A.BasicTransform):
    """Convert image and mask to `torch.Tensor`
    * Image numpy: [H, W, C] -> Image tensor: [C, H, W]
    * Mask numpy: [H, W, 1] -> Mask tensor: [1, H, W]
    """
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):
        """Image from numpy [H, W, C] to tensor [C, H, W]"""
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return torch.from_numpy(np.transpose(img, (2, 0, 1)))

    def apply_to_mask(self, mask, **params):
        """Mask from numpy [H, W] to tensor [1, H, W]"""
        # Adding channel to first dim if mask has no channel
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        # Transposing channel to channel first if mask has channel
        elif mask.ndim == 3:
            # [H, W, C] to tensor [C, H, W] in case mask has C > 1
            mask = mask.transpose(2, 0, 1)
        else:
            raise ValueError('Mask should have shape [H, W] without, '
                             'channel however provided mask shape was: '
                             '{}'.format(mask.shape))
        # To numpy
        return torch.from_numpy(mask)
