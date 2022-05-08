import numpy as np
import torch
from typing import Union
import numpy as np
from odeon.data.transforms import BasicTransform


class HWC_to_CHW(BasicTransform):
    def __init__(
        self, img_only: bool = False, mask_only: bool = False
    ):
        super(HWC_to_CHW, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only

    @staticmethod
    def swap_axes(array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        array = array.swapaxes(0, 2).swapaxes(1, 2)
        return array

    def apply_to_img(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self.swap_axes(img)

    def apply_to_mask(self, mask: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self.swap_axes(mask)


class CHW_to_HWC(BasicTransform):
    def __init__(self, img_only: bool = False, mask_only: bool = False):
        super(CHW_to_HWC, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only

    @staticmethod
    def swap_axes(array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # swap the axes order from (bands, rows, columns) to (rows, columns, bands)
        array = array.swapaxes(0, 2).swapaxes(0, 1)
        return array

    def apply_to_img(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self.swap_axes(img)

    def apply_to_mask(self, mask: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self.swap_axes(mask)



class ToDoubleTensor(object):
    """Convert ndarrays of sample(image, mask) into Tensors"""

    def __call__(self, **sample):
        image, mask = sample['image'], sample['mask']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()
        mask = mask.transpose((2, 0, 1)).copy()
        return {
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask).float()
        }


class ToSingleTensor(object):
    """Convert ndarrays of image into Tensors"""

    def __call__(self, **sample):
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()
        return torch.from_numpy(image).float()


class ToPatchTensor(object):
    """Convert ndarrays of image, scalar index, affineasndarray into Tensors"""

    def __call__(self, **sample):

        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()
        index = sample["index"]
        affine = sample["affine"]
        return {
                "image": torch.from_numpy(image).float(),
                "index": torch.from_numpy(index).int(),
                "affine": torch.from_numpy(affine).float()
                }


class ToWindowTensor(object):
    """Convert ndarrays of image, scalar index"""

    def __call__(self, **sample):

        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()
        index = sample["index"]

        return {
                "image": torch.from_numpy(image).float(),
                "index": torch.from_numpy(index).int()
                }
