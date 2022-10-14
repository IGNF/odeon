from logging import getLogger
from pathlib import Path
from typing import Dict

import albumentations as A
import geopandas as gpd
import matplotlib.pyplot as plt
# import numpy as np
import rasterio as rio
import torch
# import torch
from rasterio.plot import reshape_as_image

from odeon.core.transform import AlbuTransform

logger = getLogger(__name__)


def test_albu_transform(path_to_test_data, session_global_datadir):

    tmp_dir: Path = Path(session_global_datadir)
    print(tmp_dir)
    root_dir: Path = Path(path_to_test_data["root_dir"])
    dataset: str = path_to_test_data["zone_data"]
    gdf: gpd.GeoDataFrame = gpd.read_file(dataset)
    pipe = [A.OneOf([A.RandomResizedCrop(height=256,
                                         width=256,
                                         scale=(0.5, 1.5),
                                         p=1.0),
                     A.RandomCrop(height=256, width=256)], p=1.0),
            A.RandomRotate90(p=0.5),
            A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.75),
            A.Transpose(p=0.5)]
    input_fields: Dict = {"T-0": {"name": "T0_path", "type": "raster", "dtype": "uint8"},
                          "T-1": {"name": "t1_path", "type": "raster", "dtype": "uint8"},
                          "change_mask": {"name": "change_pat", "type": "mask", "encoding": "integer"}}
    # additional_targets = {'image': 'image', 'T-1': 'image', 'change_mask': 'mask'}
    transform = AlbuTransform(pipe=pipe,
                              input_fields=input_fields)
    # transform_2 = A.Compose(pipe, additional_targets=additional_targets)
    print(transform.additional_targets)

    for idx, row in gdf.iterrows():

        t0 = root_dir / row['T0_path']
        with rio.open(t0) as src:
            img_t0 = reshape_as_image(src.read())
        t1 = root_dir / row['t1_path']
        with rio.open(t1) as src:
            img_t1 = reshape_as_image(src.read())
        change = root_dir / row['change_pat']
        with rio.open(change) as src:
            mask = reshape_as_image(src.read())

        patch = dict()
        patch['T-0'] = img_t0
        patch['T-1'] = img_t1
        patch['change_mask'] = mask
        print(patch.keys())
        # exit(0)
        trans_patch = transform(data=patch)
        assert trans_patch['T-0'].shape == (5, 256, 256)
        assert isinstance(trans_patch['T-0'], torch.Tensor)
        assert trans_patch['T-1'].shape == (5, 256, 256)
        assert isinstance(trans_patch['T-1'], torch.Tensor)
        assert trans_patch['change_mask'].shape == (1, 256, 256)
        img_t0_trans = reshape_as_image(trans_patch['T-0'][0:3, :, :].numpy())
        img_t1_trans = reshape_as_image(trans_patch['T-1'][0:3, :, :].numpy())
        mask_trans = reshape_as_image(trans_patch['change_mask'].numpy()) * 255
        print(type(img_t0_trans))
        print(img_t0_trans.shape)
        # exit(0)

        output_file = tmp_dir / f'{idx}.png'
        fig = plt.figure(figsize=(20, 20))
        ax = []

        ax.append(fig.add_subplot(3, 3, 1))
        ax[-1].clear()
        ax[-1].set_title("image t0 transformed")
        plt.imshow(img_t0_trans)

        ax.append(fig.add_subplot(3, 3, 2))
        ax[-1].clear()
        ax[-1].set_title("image t1 transformed")
        plt.imshow(img_t1_trans)

        ax.append(fig.add_subplot(3, 3, 3))
        ax[-1].clear()
        ax[-1].set_title("mask transformed")
        # plt.imshow(trans_patch['change_mask'].numpy().astype('uint8').transpose(0, 2).transpose(0, 1) * 255)
        plt.imshow(mask_trans)

        ax.append(fig.add_subplot(3, 3, 4))
        ax[-1].clear()
        ax[-1].set_title("image t0")
        plt.imshow(img_t0[:, :, 0:3])

        ax.append(fig.add_subplot(3, 3, 5))
        ax[-1].clear()
        ax[-1].set_title("image t1")
        plt.imshow(img_t1[:, :, 0:3])

        ax.append(fig.add_subplot(3, 3, 6))
        ax[-1].clear()
        ax[-1].set_title("mask")
        plt.imshow(mask)
        plt.savefig(output_file)
        plt.close(fig)

    pipe = [A.OneOf([A.RandomResizedCrop(height=256,
                                         width=256,
                                         scale=(0.5, 1.5),
                                         p=1.0),
                     A.RandomCrop(height=256, width=256)], p=1.0),
            A.RandomRotate90(p=0.5),
            A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.75),
            A.Transpose(p=0.5)]

    # SAME test with 'image' in field
    input_fields: Dict = {"image": {"name": "T0_path", "type": "raster", "dtype": "uint8"},
                          "T-1": {"name": "t1_path", "type": "raster", "dtype": "uint8"},
                          "change_mask": {"name": "change_pat", "type": "mask", "encoding": "integer"}}
    transform = AlbuTransform(pipe=pipe,
                              input_fields=input_fields)

    # transform_2 = A.Compose(pipe, additional_targets=additional_targets)
    print(transform.additional_targets)

    for idx, row in gdf.iterrows():
        t0 = root_dir / row['T0_path']
        with rio.open(t0) as src:
            img_t0 = reshape_as_image(src.read())
        t1 = root_dir / row['t1_path']
        with rio.open(t1) as src:
            img_t1 = reshape_as_image(src.read())
        change = root_dir / row['change_pat']
        with rio.open(change) as src:
            mask = reshape_as_image(src.read())

        patch = dict()
        patch['image'] = img_t0
        patch['T-1'] = img_t1
        patch['change_mask'] = mask
        print(patch.keys())

        trans_patch = transform(data=patch)
        assert 'image' in trans_patch.keys()
        img_t0_trans = reshape_as_image(trans_patch['image'][0:3, :, :].numpy())
        img_t1_trans = reshape_as_image(trans_patch['T-1'][0:3, :, :].numpy())
        mask_trans = reshape_as_image(trans_patch['change_mask'].numpy()) * 255
        print(type(img_t0_trans))
        print(img_t0_trans.shape)

        output_file = tmp_dir / f'{idx}-with-image-field.png'
        fig = plt.figure(figsize=(20, 20))
        ax = []

        ax.append(fig.add_subplot(3, 3, 1))
        ax[-1].clear()
        ax[-1].set_title("image image transformed")
        plt.imshow(img_t0_trans)

        ax.append(fig.add_subplot(3, 3, 2))
        ax[-1].clear()
        ax[-1].set_title("image t1 transformed")
        plt.imshow(img_t1_trans)

        ax.append(fig.add_subplot(3, 3, 3))
        ax[-1].clear()
        ax[-1].set_title("mask transformed")
        # plt.imshow(trans_patch['change_mask'].numpy().astype('uint8').transpose(0, 2).transpose(0, 1) * 255)
        plt.imshow(mask_trans)

        ax.append(fig.add_subplot(3, 3, 4))
        ax[-1].clear()
        ax[-1].set_title("image image")
        plt.imshow(img_t0[:, :, 0:3])

        ax.append(fig.add_subplot(3, 3, 5))
        ax[-1].clear()
        ax[-1].set_title("image t1")
        plt.imshow(img_t1[:, :, 0:3])

        ax.append(fig.add_subplot(3, 3, 6))
        ax[-1].clear()
        ax[-1].set_title("mask")
        plt.imshow(mask)
        plt.savefig(output_file)
        plt.close(fig)
