""" package utility for image in NDArray format or Pillow format"""
import numpy as np
import torch


def img_to_tensor(img: np.ndarray):
    """

    :param img:
    :return:
    """

    return torch.from_numpy(img.transpose((2, 0, 1)))


def tensor_to_image(x: torch.Tensor):
    """

    :param x:
    :return:
    """

    return x.cpu().numpy().transpose((1, 2, 0))
