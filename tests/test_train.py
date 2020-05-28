import os
import gdal
import numpy as np
import pytest
from torch.utils.data import DataLoader

from odeon.nn.datasets import PatchDataset

class TestPatchDataset(object):

    @pytest.fixture
    def generate_image(self):

        data = np.zeros((5, 512, 512))
        data[:, int(512/2), int(512/2)] = 1
        ds = gdal.GetDriverByName('GTiff').Create("/tmp/test_image.tif", 512, 512, 5, gdal.GDT_Byte)
        for i in range(data.shape[0]):
            tmpbnd = ds.GetRasterBand(i+1)
            tmpbnd.WriteArray(data[i, :, :], 0, 0)
        ds.FlushCache()
        ds = None

        yield

        os.remove("/tmp/test_image.tif")

    @pytest.fixture
    def generate_mask(self):

        data = np.zeros((9, 512, 512))
        data[:, int(512/2), int(512/2)] = 1
        ds = gdal.GetDriverByName('GTiff').Create("/tmp/test_mask.tif", 512, 512, 9, gdal.GDT_Byte)
        for i in range(data.shape[0]):
            tmpbnd = ds.GetRasterBand(i+1)
            tmpbnd.WriteArray(data[i, :, :], 0, 0)
        ds.FlushCache()
        ds = None

        yield

        os.remove("/tmp/test_mask.tif")

    def test_crop(self, generate_image, generate_mask):

        image_files = ['/tmp/test_image.tif']
        mask_files = ['/tmp/test_mask.tif']
        dataset_optional_args = {'width': 100}
        dataset = PatchDataset(image_files, mask_files, **dataset_optional_args)

        dataloader = DataLoader(dataset)

        sample = next(iter(dataloader))

        # assert output size is correct
        assert sample['image'].shape[1] == 100
        assert sample['image'].shape[2] == 100
        assert sample['mask'].shape[1] == 100
        assert sample['mask'].shape[2] == 100

        # assert center pixel is correct
        center_x = int(sample['image'].shape[1]/2)
        center_y = int(sample['image'].shape[2]/2)
        assert sample['image'][0, center_x, center_y, :].numpy().tolist() == [1/255, 1/255, 1/255, 1/255, 1/255]


class TestTransformation(object):

    def test_transforms(self):
        pass