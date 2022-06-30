from typing import List, Optional, Dict
import torch
import numpy as np
import albumentations as A


class SegmentationTaskDataAugmentation:

    def __init__(self, mean, std, pipe: List = None):

        self.mean = mean
        self.std = std
        self.pipe = pipe if pipe is not None else []
        self.normalize = None

        if (self.mean is not None) and (self.std is not None):

            self.normalize = A.Normalize(mean=mean, std=std)

        pipe.append(ToTensorCustom())
        self.transform = A.Compose(pipe)

    def forward(self, img, mask, *args, **kwargs):

        img = self.normalize(image=img)["image"] if self.normalize is not None else img
        transformed = self.transform(image=img, mask=mask)
        return transformed

    def __call__(self, img, mask, *args, **kwargs):

        return self.forward(img, mask)


class SegmentationTaskTestDataAugmentation(SegmentationTaskDataAugmentation):

    def __init__(self, mean, std, pipe: List = None):

        super(SegmentationTaskTestDataAugmentation, self).__init__(mean, std, pipe)
        self.additional_targets = {
            "img_2016_style_2016": "image",
            "img_2016_style_2019": "image",
            "mask_2016": "mask",
            "mask_change": "mask"}
        self.transform = A.Compose(pipe, additional_targets=self.additional_targets)

    def forward(self,
                img,
                mask,
                img_2016_style_2016=None,
                img_2016_style_2019=None,
                mask_2016=None,
                mask_change=None):

        img = self.normalize(image=img)["image"] if self.normalize is not None else img
        img_2016_style_2016 = self.normalize(image=img_2016_style_2016)["image"]\
            if self.normalize is not None else img_2016_style_2016
        img_2016_style_2019 = self.normalize(image=img_2016_style_2019)["image"]\
            if self.normalize is not None else img_2016_style_2019

        transformed = self.transform(image=img,
                                     img_2016_style_2016=img_2016_style_2016,
                                     img_2016_style_2019=img_2016_style_2019,
                                     mask=mask,
                                     mask_2016=mask_2016,
                                     mask_change=mask_change)

        return transformed

    def __call__(self,
                 img,
                 mask,
                 img_2016_style_2016=None,
                 img_2016_style_2019=None,
                 mask_2016=None,
                 mask_change=None):

        return self.forward(img,
                            mask,
                            img_2016_style_2016,
                            img_2016_style_2019,
                            mask_2016,
                            mask_change)


class DeNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        return tensor


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
        return torch.from_numpy(img.transpose(2, 0, 1))

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