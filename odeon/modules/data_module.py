import os
import csv
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
from odeon import LOGGER
from odeon.nn.datasets import PatchDataset
from odeon.commons.guard import check_files, check_raster_bands
from odeon.commons.exception import OdeonError, ErrorCodes

RANDOM_SEED = 42
BATCH_SIZE = 5
NUM_WORKERS = 4
PERCENTAGE_VAL = 0.3


class SegDataModule(LightningDataModule):

    def __init__(self,
                 train_file,
                 val_file=None,
                 image_bands=None,
                 mask_bands=None,
                 transforms=None,
                 width=None,
                 height=None,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS,
                 percentage_val=PERCENTAGE_VAL,
                 pin_memory=True,
                 deterministic=False,
                 subset=False):

        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.image_bands = image_bands
        self.mask_bands = mask_bands
        self.transforms = transforms
        self.width = width
        self.height = height
        self.num_workers = num_workers
        self.percentage_val = percentage_val
        self.pin_memory = pin_memory
        self.subset = subset
        if deterministic:
            self.random_seed = None
            self.shuffle = False
        else:
            self.random_seed = RANDOM_SEED
            self.shuffle = True
        self.train_image_files, self.val_image_files, self.train_mask_files, self.val_mask_files = \
            self.get_split_files()
        self.sample_dims = self.get_dims()
        self.image_bands, self.mask_bands = self.get_bands(image_bands, mask_bands)
        self.num_classes = len(self.mask_bands)
        self.num_channels = len(self.image_bands)
        self.train_batch_size, self.val_batch_size = self.get_batch_size(batch_size)
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        pass

    def setup(self, stage=None):        
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.train_dataset = PatchDataset(image_files=self.train_image_files,
                                              mask_files=self.train_mask_files,
                                              transform=self.transforms['train'],
                                              image_bands=self.image_bands,
                                              mask_bands=self.mask_bands,
                                              width=self.width,
                                              height=self.height)

            self.val_dataset = PatchDataset(image_files=self.val_image_files,
                                             mask_files=self.val_mask_files,
                                             transform=self.transforms['val'],
                                             image_bands=self.image_bands,
                                             mask_bands=self.mask_bands,
                                             width=self.width,
                                             height=self.height)
            if self.subset is True:
                self.train_dataset = Subset(self.train_dataset, range(0, 20))
                self.val_dataset = Subset(self.val_dataset, range(0, 10))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)

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
        return train_image_files, val_image_files, train_mask_files, val_mask_files

    @staticmethod
    def get_samples(image_files, mask_files):
        image_file, mask_file = image_files[0], mask_files[0]
        with rasterio.open(image_file) as image_raster:
            image = image_raster.read().swapaxes(0, 2).swapaxes(0, 1)
        with rasterio.open(mask_file) as mask_raster:
            mask = mask_raster.read().swapaxes(0, 2).swapaxes(0, 1)
        return {'image': image, 'mask': mask}

    @staticmethod
    def check_sample(sample):
        if sample['image'].shape[0:-1] != sample['mask'].shape[0:-1]:
            LOGGER.error('ERROR: check the width/height of the inputs masks and detections. \
                Those input data should have the same width/height.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "Detections and masks have different width/height.")

    def get_dims(self):
        train_sample  = self.get_samples(self.train_image_files,
                                         self.train_mask_files)
        val_sample  = self.get_samples(self.val_image_files,
                                       self.val_mask_files)
        self.check_sample(train_sample)
        self.check_sample(val_sample)
        if train_sample['image'].shape != val_sample['image'].shape:
            LOGGER.warning("WARNING: Data in train dataset and validation dataset don\'t have\
                           the same dimensions.")
        return {'image': train_sample['image'].shape, 'mask': train_sample['mask'].shape}

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
        elif isinstance(parameter_batch_size, (tuple, list, np.ndarray)):
            train_batch_size = parameter_batch_size[0]
            val_batch_size = parameter_batch_size[1]
        else:
            LOGGER.error("ERROR: Parameter batch_size should a list/tuple of length one, two or three.")
            raise ValueError('Parameter batch_size is not correct.')
        assert train_batch_size <= len(self.train_image_files),\
            "batch_size must be lower than the number of files in the dataset"
        assert val_batch_size <= len(self.val_image_files),\
         "batch_size must be lower than the number of files in the dataset"                                          
        return train_batch_size, val_batch_size

    def teardown(self, stage=None):
        # Used to clean-up when the run is finished
        pass
