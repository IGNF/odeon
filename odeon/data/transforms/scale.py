import numpy as np
from odeon.data.transforms import BasicTransform 


class ScaleImageToFloat(BasicTransform):
    """
    scale an input sample to float image between [0, 1]
    """

    def __init__(
        self,
        scale_factor: float = 255,
        clip: bool = False,
        img_only: bool = True,
        mask_only: bool = False,
    ):
        super(ScaleImageToFloat, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only
        self.scale_factor = scale_factor
        self.clip = clip

    def _scale_array(self, arr: np.ndarray) -> np.ndarray:
        arr = np.multiply(arr, 1.0 / self.scale_factor, dtype=np.float32)
        if self.clip:
            return np.clip(arr, 0, 1)
        else:
            return arr

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        return self._scale_array(img)

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        return self._scale_array(mask)


class FloatImageToByte(BasicTransform):
    """
    scale an input image from [0-1] to [0-255] mainly ofr rgb display purpose
    """

    def __init__(
        self, clip: bool = False, img_only: bool = True, mask_only: bool = False
    ):
        super(FloatImageToByte, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only
        self.scale_factor = 255
        self.clip = clip

    def _float_to_byte(self, arr: np.ndarray) -> np.ndarray:
        arr = np.multiply(arr, self.scale_factor, dtype=np.float32)
        arr = arr.astype(np.uint8)
        if self.clip:
            return np.clip(arr, 0, 255)
        else:
            return arr

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        return self._float_to_byte(img)

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        return self._float_to_byte(mask)


