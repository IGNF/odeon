from typing import Callable, Dict, List, Optional

import albumentations as A
import numpy as np
import torch

from odeon.core.data import IMAGE_MODALITY


class AlbuTransform:

    def __init__(self,
                 input_fields: Dict,
                 pipe: Optional[List[Callable]] = None
                 ):

        self._input_fields = input_fields
        self._pipe: List = list() if pipe is None else pipe
        self._pipe.append(ToTensorCustom())
        self._additional_targets: Dict = dict()
        self._has_image: bool = False
        self._key_image: str = ''
        self._input_fields_has_image_field: bool = sum([1 if key == 'image' else 0
                                                        for key in self._input_fields.keys()]) > 0
        self._image_type_fields = {key: value for key, value in self._input_fields.items() if value["type"]
                                   in IMAGE_MODALITY}
        if self._input_fields_has_image_field:

            # self._additional_targets['image'] = 'image'
            self._key_image = 'image'

        else:

            self._key_image = next(iter(self._image_type_fields.keys()))

        for key, value in self._input_fields.items():

            if value['type'] in IMAGE_MODALITY:
                if key != 'image':
                    self._additional_targets[key] = 'image'

            if value['type'] == 'mask':
                self._additional_targets[key] = 'mask'

        self.transform: A.Compose = A.Compose(self._pipe, additional_targets=self._additional_targets)

    @property
    def additional_targets(self):
        return self._additional_targets

    def __call__(self, data: Dict, *args, **kwargs):

        """
        transform_data: Dict = dict({'image': data[self._key_image]})
        del data[self._key_image]
        transform_image: Dict = {key: value for key, value in data.items() if key in self._additional_targets}
        transform_data.update(transform_image)
        other_data = {key: value for key, value in data.items() if key not in self._additional_targets}
        """

        additional_transform: Dict = dict()
        other_data: Dict = dict()
        image_value = data[self._key_image]

        for key, value in data.items():
            if key == self._key_image:
                pass
            elif (key in self.additional_targets) and (key != self._key_image):
                additional_transform[key] = value
            else:
                other_data[key] = value
        """
        print(f'transform data: {additional_transform.keys()}')
        for key, value in additional_transform.items():
            print(f'key: {key}, type: {type(value)}')
        """
        transform_data = self.transform(image=image_value, **additional_transform)
        if self._key_image != 'image':
            transform_data[self._key_image] = transform_data['image']
            del transform_data['image']
        return dict(transform_data, **other_data)


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
        """
        Image from numpy [H, W, C] to tensor [C, H, W]
        Parameters
        ----------
        img
        params

        Returns
        -------

        """

        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return torch.from_numpy(np.transpose(img, (2, 0, 1)))

    def apply_to_mask(self, mask, **params):
        """
        Mask from numpy [H, W] to tensor [1, H, W]
        Parameters
        ----------
        mask
        params

        Returns
        -------

        """
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
