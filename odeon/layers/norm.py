from typing import List, Union

import numpy as np
from layers.core.data import DTYPE_MAX, InputDType
from torch import Tensor

MEAN_DEFAULT_VALUE = 0.5
STD_DEFAULT_VALUE = 0.5


def normalize(img: np.ndarray,
              mean: List[float],
              std: List[float],
              max_value: Union[float | int] = float(DTYPE_MAX[InputDType.UINT8.value]),
              min_value: Union[float | int] = 0) -> np.ndarray:
    """
    Normalize an image with mean and std
    Parameters
    ----------
    img : np.ndarray
    mean : List[float]
    std : List[float]
    max_value : float | int
    min_value : float | int
    Returns
    -------

    """
    if min_value != 0:
        img = (img - min_value) * max_value
    max_value = float(max_value)
    mean_np = np.array(mean, dtype=np.float32)
    mean_np *= max_value
    std_np = np.array(std, dtype=np.float32)
    std_np *= max_value
    denominator = np.reciprocal(std_np, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean_np
    img *= denominator
    return img


def denormalize_tensor(image: Tensor, mean: List[float], std: List[float]):
    """
    Denormalize a tensor image with mean and standard deviation.
    Parameters
    ----------
    image : Tensor
    mean : List[float]
    std : List[float]

    Returns
    -------
    Tensor
    """
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)

    return image
