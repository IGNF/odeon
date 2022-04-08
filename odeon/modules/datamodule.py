import os
import csv
import numpy as np
import rasterio
from typing import Optional, Union, List
from albumentations import Compose
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
from odeon import LOGGER
from odeon.nn.datasets import PatchDataset
from odeon.commons.guard import check_files, check_raster_bands
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.configs.core import DataModuleConf, TransformsConf

RANDOM_SEED = 42


class SegDataModule(LightningDataModule):

    def __init__(
        self,
        config: DataModuleConf,
        transform_config: TransformsConf
        )-> None:
        
        super().__init__()
        self.__dict__.update(**config)  # save_hyperparameters() coul be used instead, not sure it will be more useful

        if self.deterministic:
            self.random_seed = None
            self.shuffle = False
        else:
            self.random_seed = RANDOM_SEED if self.seed is None else self.seed
            self.shuffle = True

        self.train_image_files, self.val_image_files, self.test_image_files = None, None, None
        self.train_mask_files, self.val_mask_files, self.test_mask_files = None, None, None
        self.train_batch_size, self.val_batch_size, self.test_batch_size = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.resolution, self.meta = {}, {}

        self.get_split_files()
        self.sample_dims = self.get_dims_and_meta()
        if self.resolution is not None:
            self.get_resolution(self.resolution)
        self.image_bands, self.mask_bands = self.get_bands(self.image_bands, self.mask_bands)
        self.num_classes = len(self.mask_bands)
        self.num_channels = len(self.image_bands)
        self.get_batch_size(self.batch_size)

        # Init transformations sequences
        self.train_tfm = self.instantiate_transforms(transform_config.train)
        self.val_tfm = self.instantiate_transforms(transform_config.val)
        if config.test_file is not None and transform_config.test is not None:
            self.test_tfm = self.instantiate_transforms(transform_config.test)
        else:
            self.test_tfm = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            if not self.train_dataset and not self.val_dataset:
                self.train_dataset = PatchDataset(image_files=self.train_image_files,
                                                  mask_files=self.train_mask_files,
                                                  transform=self.train_tfm,
                                                  image_bands=self.image_bands,
                                                  mask_bands=self.mask_bands,
                                                  width=self.width,
                                                  height=self.height)

                self.val_dataset = PatchDataset(image_files=self.val_image_files,
                                                mask_files=self.val_mask_files,
                                                transform=self.val_tfm,
                                                image_bands=self.image_bands,
                                                mask_bands=self.mask_bands,
                                                width=self.width,
                                                height=self.height)

                if self.subset is True:
                    self.train_dataset = Subset(self.train_dataset, range(0, 20))
                    self.val_dataset = Subset(self.val_dataset, range(0, 10))

        elif stage == "test" or stage == "predict":
            if not self.test_dataset:
                self.test_dataset = PatchDataset(image_files=self.test_image_files,
                                                 mask_files=self.test_mask_files,
                                                 transform=self.test_tfm,
                                                 image_bands=self.image_bands,
                                                 mask_bands=self.mask_bands,
                                                 width=self.width,
                                                 height=self.height,
                                                 get_sample_info=self.get_sample_info)
                if self.subset is True:
                    self.test_dataset = Subset(self.test_dataset, range(0, 10))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False,
                          drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False,
                          drop_last=self.drop_last)

    def predict_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False,
                          drop_last=self.drop_last)

    def read_csv_sample_file(self, file_path):
        image_files = []
        mask_files = []
        if not os.path.exists(file_path):
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"file ${file_path} does not exist.")
        with open(file_path) as csvfile:
            sample_reader = csv.reader(csvfile)
            for item in sample_reader:
                image_files.append(item[0])
                mask_files.append(item[1])
        return image_files, mask_files

    def get_split_files(self):
        # Read csv file with columns: image, mask
        if self.train_file:
            train_image_files, train_mask_files = self.read_csv_sample_file(self.train_file)
            if self.val_file is not None:
                val_image_files, val_mask_files = self.read_csv_sample_file(self.val_file)
            else:
                train_image_files, val_image_files, train_mask_files, val_mask_files = \
                    train_test_split(train_image_files,
                                     train_mask_files,
                                     test_size=self.percentage_val,
                                     random_state=self.random_seed)
            for list_files in [train_image_files, val_image_files, train_mask_files, val_mask_files]:
                check_files(list_files)
            self.train_image_files, self.train_mask_files = train_image_files, train_mask_files
            self.val_image_files, self.val_mask_files = val_image_files, val_mask_files

        if self.test_file:
            test_image_files, test_mask_files = self.read_csv_sample_file(self.test_file)
            for list_files in [test_image_files, test_mask_files]:
                check_files(list_files)
            self.test_image_files, self.test_mask_files = test_image_files, test_mask_files

    @staticmethod
    def get_samples(image_files, mask_files, index=0):
        image_file, mask_file = image_files[index], mask_files[index]
        with rasterio.open(image_file) as image_raster:
            image = image_raster.read().swapaxes(0, 2).swapaxes(0, 1)
            meta = image_raster.meta
            res = image_raster.res
        with rasterio.open(mask_file) as mask_raster:
            mask = mask_raster.read().swapaxes(0, 2).swapaxes(0, 1)
        return {"image": image, "mask": mask, "meta": meta, "resolution": res}

    @staticmethod
    def check_sample(sample):
        if sample['image'].shape[0:-1] != sample['mask'].shape[0:-1]:
            LOGGER.error('ERROR: check the width/height of the inputs masks and detections. \
                Those input data should have the same width/height.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "Detections and masks have different width/height.")

    def get_dims_and_meta(self):
        if self.train_file:
            train_sample = self.get_samples(self.train_image_files,
                                            self.train_mask_files)
            val_sample = self.get_samples(self.val_image_files,
                                          self.val_mask_files)
            self.check_sample(train_sample)
            self.check_sample(val_sample)

            if train_sample['image'].shape != val_sample['image'].shape:
                LOGGER.warning("WARNING: Data in train dataset and validation dataset don\'t have\
                            the same dimensions.")

            # Meta and resolution definition
            self.resolution["train"] = train_sample["resolution"]
            self.resolution["val"] = val_sample["resolution"]

            self.meta["train"] = train_sample["meta"]
            self.meta["val"] = val_sample["meta"]

        if self.test_file:
            test_sample = self.get_samples(self.test_image_files,
                                           self.test_mask_files)
            self.check_sample(test_sample)
            self.resolution["test"] = test_sample["resolution"]
            self.meta["test"] = test_sample["meta"]
        else:
            self.resolution["test"] = self.resolution["val"]
            self.meta["test"] = self.meta["val"]

        if self.train_file:
            dims = {'image': train_sample['image'].shape, 'mask': train_sample['mask'].shape}
        elif self.test_file:
            dims = {'image': test_sample['image'].shape, 'mask': test_sample['mask'].shape}
        else:
            LOGGER.error("ERROR: SegDataModule need at least a train file or a test file to be instantiate.")
            raise ValueError('Parameter train_file/test_file is not correct.')

        return dims

    def get_bands(self, image_bands, mask_bands):
        if image_bands is None:
            image_bands = np.arange(self.sample_dims['image'][-1])
        else:
            check_raster_bands(raster_band=np.arange(1, self.sample_dims['image'][-1] + 1),
                               proposed_bands=image_bands)
        if mask_bands is None:
            mask_bands = np.arange(self.sample_dims['mask'][-1])
        else:
            check_raster_bands(raster_band=np.arange(1, self.sample_dims['mask'][-1] + 1),
                               proposed_bands=mask_bands)
        return image_bands, mask_bands

    def get_batch_size(self, parameter_batch_size):
        if isinstance(parameter_batch_size, int):
            train_batch_size = parameter_batch_size
            val_batch_size = parameter_batch_size
            test_batch_size = parameter_batch_size
        elif isinstance(parameter_batch_size, (tuple, list, np.ndarray)):
            train_batch_size = parameter_batch_size[0]
            val_batch_size = parameter_batch_size[1]
            test_batch_size = parameter_batch_size[-1]
        else:
            LOGGER.error("ERROR: Parameter batch_size should a list/tuple of length one, two or three.")
            raise ValueError('Parameter batch_size is not correct.')

        if self.train_file:
            assert train_batch_size <= len(self.train_image_files),\
                "batch_size must be lower than the number of files in the dataset"
            assert val_batch_size <= len(self.val_image_files),\
            "batch_size must be lower than the number of files in the dataset"

        if self.test_file is not None:
            assert test_batch_size <= len(self.test_image_files),\
            "batch_size must be lower than the number of files in the dataset"

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def get_resolution(self, parameter_resolution):
        if isinstance(parameter_resolution, int):
            self.resolution["train"] = [parameter_resolution, parameter_resolution]
            self.resolution["val"]  = [parameter_resolution, parameter_resolution]
            self.resolution["test"]  = [parameter_resolution, parameter_resolution]
        elif isinstance(parameter_resolution, (tuple, list, np.ndarray)):
            self.resolution["train"]  = parameter_resolution[0]
            self.resolution["val"]  = parameter_resolution[1]
            self.resolution["test"]  = parameter_resolution[-1]

    def teardown(self, stage=None):
        # Used to clean-up when the run is finished
        pass

    @staticmethod
    def instantiate_transforms(transforms: Optional[List])-> Compose:
        if transforms is None:
            return None
        return Compose([instantiate(transform) for transform in transforms])
