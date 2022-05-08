import numpy as np
from odeon.data.transforms import BasicTransform


class DeNormalize(BasicTransform):
    def __init__(
        self,
        mean: list[float],
        std: list[float],
        max_pixel_value: float = 255.0,
        img_only: bool = True,
        mask_only: bool = False,
    ):
        super(DeNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.img_only = img_only
        self.mask_only = mask_only
        self.max_pixel_value = max_pixel_value

    def _denormalize(self, img: np.ndarray) -> np.ndarray:
        mean = np.array(self.mean, dtype=np.float32)
        mean *= self.max_pixel_value

        std = np.array(self.std, dtype=np.float32)
        std *= self.max_pixel_value

        img = img.astype(np.float32)
        img *= std
        img += mean
        return img
    
    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        return self._denormalize(img)

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        return self._denormalize(mask)