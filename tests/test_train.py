import os
import gdal
import numpy as np
import pytest
import random

from torch.utils.data import DataLoader

from odeon.nn.datasets import PatchDataset
from odeon.nn.transforms import Rotation90, ToDoubleTensor, Compose

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class TestPatchDataset(object):

    @pytest.fixture
    def generate_sample(self):

        data = np.zeros((5, 512, 512))
        data[:, int(512/2), int(512/2)] = 1
        ds = gdal.GetDriverByName('GTiff').Create("/tmp/test_image.tif", 512, 512, 5, gdal.GDT_Byte)
        for i in range(data.shape[0]):
            tmpbnd = ds.GetRasterBand(i+1)
            tmpbnd.WriteArray(data[i, :, :], 0, 0)
        ds.FlushCache()
        ds = None

        data = np.zeros((9, 512, 512))
        data[:, int(512/2), int(512/2)] = 1
        ds = gdal.GetDriverByName('GTiff').Create("/tmp/test_mask.tif", 512, 512, 9, gdal.GDT_Byte)
        for i in range(data.shape[0]):
            tmpbnd = ds.GetRasterBand(i+1)
            tmpbnd.WriteArray(data[i, :, :], 0, 0)
        ds.FlushCache()
        ds = None

        yield

        os.remove("/tmp/test_image.tif")
        os.remove("/tmp/test_mask.tif")

    def test_crop(self, generate_sample):

        image_files = ['/tmp/test_image.tif']
        mask_files = ['/tmp/test_mask.tif']
        dataset_optional_args = {'width': 100}
        dataset = PatchDataset(image_files, mask_files, **dataset_optional_args)

        dataloader = DataLoader(dataset)

        sample = next(iter(dataloader))  # B x C x W x H

        # assert output size is correct
        assert sample['image'].shape[2] == 100
        assert sample['image'].shape[3] == 100
        assert sample['mask'].shape[2] == 100
        assert sample['mask'].shape[3] == 100

        # assert center pixel is correct
        center_x = int(sample['image'].shape[2]/2)
        center_y = int(sample['image'].shape[3]/2)
        np.testing.assert_array_equal(
            sample['image'][0, :, center_x, center_y].numpy(),
            1/255
        )


class TestTransformation(object):

    @pytest.fixture
    def generate_sample(self):

        data = np.zeros((5, 512, 512))
        data[:, int(512/2) - 1, int(512/2) - 1] = 1
        data[:, int(512/2), int(512/2) - 1] = 2
        data[:, int(512/2), int(512/2)] = 3
        data[:, int(512/2) - 1, int(512/2)] = 4

        ds = gdal.GetDriverByName('GTiff').Create("/tmp/test_image.tif", 512, 512, 5, gdal.GDT_Byte)
        for i in range(data.shape[0]):
            tmpbnd = ds.GetRasterBand(i+1)
            tmpbnd.WriteArray(data[i, :, :], 0, 0)
        ds.FlushCache()
        ds = None

        data = np.zeros((9, 512, 512))
        data[:, int(512/2) - 1, int(512/2) - 1] = 1
        data[:, int(512/2), int(512/2) - 1] = 2
        data[:, int(512/2), int(512/2)] = 3
        data[:, int(512/2) - 1, int(512/2)] = 4
        ds = gdal.GetDriverByName('GTiff').Create("/tmp/test_mask.tif", 512, 512, 9, gdal.GDT_Byte)
        for i in range(data.shape[0]):
            tmpbnd = ds.GetRasterBand(i+1)
            tmpbnd.WriteArray(data[i, :, :], 0, 0)
        ds.FlushCache()
        ds = None

        yield

        os.remove("/tmp/test_image.tif")
        os.remove("/tmp/test_mask.tif")

    def test_transforms(self, generate_sample):

        image_files = ['/tmp/test_image.tif']
        mask_files = ['/tmp/test_mask.tif']

        random.seed(2020)
        dataset = PatchDataset(image_files, mask_files, transform=Compose([Rotation90(), ToDoubleTensor()]))

        dataloader = DataLoader(dataset)

        sample = next(iter(dataloader))

        # Rotation90 does a rotation of +90 degrees
        np.testing.assert_array_equal(
            sample['image'][0, :, int(512/2) - 1, int(512/2) - 1].numpy(),
            4/255
        )
        np.testing.assert_array_equal(
            sample['image'][0, :, int(512/2), int(512/2) - 1].numpy(),
            1/255
        )
        np.testing.assert_array_equal(
            sample['image'][0, :, int(512/2), int(512/2)].numpy(),
            2/255
        )
        np.testing.assert_array_equal(
            sample['image'][0, :, int(512/2) - 1, int(512/2)].numpy(),
            3/255
        )

    def test_albumentation_transforms(self, generate_sample):

        image_files = ['/tmp/test_image.tif']
        mask_files = ['/tmp/test_mask.tif']

        random.seed(2020)
        dataset = PatchDataset(image_files, mask_files, transform=A.Compose([A.RandomRotate90(), ToTensorV2()]))

        dataloader = DataLoader(dataset)

        sample = next(iter(dataloader))

        # RandomRotate90 does a rotation of -90 degrees
        np.testing.assert_array_equal(
            sample['image'][0, :, int(512/2) - 1, int(512/2) - 1].numpy(),
            2/255
        )
        np.testing.assert_array_equal(
            sample['image'][0, :, int(512/2), int(512/2) - 1].numpy(),
            3/255
        )
        np.testing.assert_array_equal(
            sample['image'][0, :, int(512/2), int(512/2)].numpy(),
            4/255
        )
        np.testing.assert_array_equal(
            sample['image'][0, :, int(512/2) - 1, int(512/2)].numpy(),
            1/255
        )

