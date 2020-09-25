import random

import numpy as np
import torch
from skimage import exposure
from skimage.color import hsv2rgb
from skimage.color import rgb2hsv
from skimage.transform import rotate
from skimage.util import random_noise

# TODO :
#   - normalisation sur les canaux RGB Irc ?
#       https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#   - investiguer les fonctions pytorch et la bibliothèque albumentation
#       https://albumentations.readthedocs.io/en/latest/index.html


class Rotation90(object):
    """Apply rotation (0, 90, 180 or 270) to image and mask, this is the default minimum transformation."""

    def __call__(self, **sample):
        image, mask = sample['image'], sample['mask']

        k = random.randint(0, 3)  # number of rotations

        # rotation
        image = np.rot90(image, k, (0, 1))
        mask = np.rot90(mask, k, (0, 1))

        return {'image': image, 'mask': mask}


class Rotation(object):
    """Apply any rotation to image and mask"""

    def __call__(self, **sample):
        image, mask = sample['image'], sample['mask']

        # rotation angle in degrees in counter-clockwise direction.
        angle = random.randint(0, 359)
        image = rotate(image, angle=angle)
        mask = rotate(mask, angle)

        return {'image': image, 'mask': mask}


class Radiometry(object):
    """Apply gamma, hue variation and noise to image and mask
       There is a 50% that each effect is applied"""

    def __call__(self, **sample):

        image, mask = sample['image'], sample['mask']

        if random.randint(0, 1):
            # gamma correction on the RGB IRC channels
            gamma_factor = random.uniform(0.5, 2.2)
            image[:, :, 0:4] = exposure.adjust_gamma(image[:, :, 0:4], gamma=gamma_factor, gain=1)

        if random.randint(0, 1):
            # allows up to 10% of color variation
            hue_factor = random.uniform(0, 0.066)

            # the rvb->hsv translation may lead to 'divide by zero' errors
            np.seterr(invalid='ignore', divide='ignore')
            # in HSV space, only the first band is modified (hue)
            hsv_img = rgb2hsv(image[:, :, 0:3])
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_factor) % 1
            image[:, :, 0:3] = hsv2rgb(hsv_img)

        if random.randint(0, 1):
            var = random.uniform(0.001, 0.01)
            image[:, :, 0:4] = random_noise(image[:, :, 0:4], var=var)

        return {'image': image, 'mask': mask}


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


class Compose(object):
    """Compose function differs from torchvision Compose as sample argument is passed unpacked to match albumentation
    behaviour.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **sample):
        for t in self.transforms:
            sample = t(**sample)
        return sample
