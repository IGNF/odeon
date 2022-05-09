from typing import Any, Dict

import numpy as np


class BasicTransform:
    def __init__(self):
        self.params: Dict[Any, Any] = {}
        self.img_only: bool = False
        self.mask_only: bool = False

    def __call__(
        self, image: np.ndarray = None, mask: np.ndarray = None
    ) -> Dict[str, Any]:
        if self.img_only:
            return {"image": self.apply_to_img(image), "mask": mask}
        elif self.mask_only:
            return {"image": image, "mask": self.apply_to_mask(mask)}
        else:
            return {"image": self.apply_to_img(image), "mask": self.apply_to_mask(mask)}

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError


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
