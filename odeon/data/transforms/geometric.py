import random

import numpy as np
from skimage.transform import rotate


class Rotation90(object):
    """Apply rotation (0, 90, 180 or 270) to image and mask, this is the default minimum transformation."""

    def __call__(self, **sample):
        image, mask = sample["image"], sample["mask"]

        k = random.randint(0, 3)  # number of rotations

        # rotation
        image = np.rot90(image, k, (0, 1))
        mask = np.rot90(mask, k, (0, 1))

        return {"image": image, "mask": mask}


class Rotation(object):
    """Apply any rotation to image and mask"""

    def __call__(self, **sample):
        image, mask = sample["image"], sample["mask"]

        # rotation angle in degrees in counter-clockwise direction.
        angle = random.randint(0, 359)
        image = rotate(image, angle=angle)
        mask = rotate(mask, angle)

        return {"image": image, "mask": mask}
