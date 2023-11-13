import os
import sys
import os.path
import csv
from tqdm import tqdm
import argparse
from pprint import pformat
import random

import numpy as np
from skimage.transform import rotate

import torch
from torch.utils.data import DataLoader

from odeon.nn.datasets import PatchDataset
from odeon.commons.json_interpreter import JsonInterpreter
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes


class Rotation90(object):
    """Apply rotation (0, 90, 180 or 270) to image and mask, this is the default minimum transformation."""

    def __call__(self, **sample):
        image, mask = sample['image'], sample['mask']

        k = random.randint(0, 3)  # number of rotations

        # rotation
        image_90 = np.rot90(image.copy(), k, (0, 1))
        mask_90 = np.rot90(mask.copy(), k, (0, 1))

        return {'image': image_90, 'mask': mask_90}

class Rotation(object):
    """Apply any rotation to image and mask"""

    def __call__(self, **sample):
        image, mask = sample['image'], sample['mask']

        # rotation angle in degrees in counter-clockwise direction.
        angle = random.randint(0, 359)
        image = rotate(image, angle=angle)
        mask = rotate(mask, angle)

        return {'image': image, 'mask': mask}


class ToDoubleTensor(object):
    """Convert ndarrays of sample(image, mask) into Tensors"""

    def __call__(self, **sample):
        image, mask = sample['image'], sample['mask']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()
        mask = mask.transpose((2, 0, 1)).copy()
        return {
            'image': torch.as_tensor(image, dtype=torch.float),
            'mask': torch.as_tensor(mask, dtype=torch.float)
        }


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


class TrainerTest():

    def __init__(
        self, verbosity, train_file, image_bands=None, mask_bands=None,
        epochs=300, batch_size=16, data_augmentation=None ):
        
        self.verbosity = verbosity
        self.batch_size = batch_size
        self.epochs = epochs

        # read csv file with columns: image, mask
        self.train_image_files, self.train_mask_files = read_csv_sample_file(train_file)

        transform_func = get_transform_func(data_augmentation)

        train_dataset = PatchDataset(
            self.train_image_files, self.train_mask_files, transform=transform_func,
            image_bands=image_bands, mask_bands=mask_bands)
        self.train_dataloader = DataLoader(
            train_dataset, self.batch_size, shuffle=True, num_workers=8, drop_last=True)


def dataloader_epochs_loop(epochs, dataloader, device):
    epoch_start = 0
    # training loop
    for epoch in range(epoch_start, epochs):
        epoch_counter = epoch
        # run a pass on current epoch
        with tqdm(total=len(dataloader),
                desc=f"Epochs {epoch_counter + 1}/{epochs}",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]') as pbar:
                
            for sample in dataloader:
                images = sample['image'].cuda(device) if device.startswith('cuda') else sample['image']
                masks = sample['mask'].cuda(device) if device.startswith('cuda') else sample['mask']
                pbar.update(1)


def get_transform_func(data_augmentation):
    
    #transformations
    if data_augmentation is None:
        data_augmentation = ['rotation90']
    transformation_dict = {
        "rotation90": Rotation90(),
        "rotation": Rotation(),
    }
    transformation_conf = data_augmentation
    transformation_keys = transformation_conf if isinstance(transformation_conf, list) else [transformation_conf]

    #transformation_functions = [
    #    value for key, value in transformation_dict.items() if key in transformation_keys
    #    ]
    #transformation_functions = list({
    #    value for key, value in transformation_dict.items() if key in transformation_keys
    #   })
    #transformation_functions.append(ToDoubleTensor())
    #transform_func = Compose(transformation_functions)

    ## A no leak
    #transform_func = Compose([ToDoubleTensor()])
    
    ## B no leak ? or very small
    #transform_func = Compose([
    #    Rotation(),
    #    ToDoubleTensor()
    #    ])
    
    transformation_functions = [
        transformation_dict["rotation90"]
    ]
    transformation_functions.append(ToDoubleTensor())
    transform_func = Compose(transformation_functions)
    return transform_func


def read_csv_sample_file(file_path):
    """Read a sample CSV file and return a list of image files and a list of mask files.
    CSV file should contain image pathes in the first column and mask pathes in the second.

    Parameters
    ----------
    file_path : str
        path to sample CSV file

    Returns
    -------
    Tuple[list, list]
        a list of image pathes and a list of mask pathes
    """
    image_files = []
    mask_files = []
    with open(file_path) as csvfile:
        sample_reader = csv.reader(csvfile)
        for item in sample_reader:
            image_files.append(item[0])
            mask_files.append(item[1])
    return image_files, mask_files


def dataloader_init(
    verbosity, train_file, image_bands=None, mask_bands=None, epochs=300,
    batch_size=16, num_workers=8, data_augmentation=None):
    
    verbosity = verbosity
    # read csv file with columns: image, mask
    train_image_files, train_mask_files = read_csv_sample_file(train_file)

    transform_func = get_transform_func(data_augmentation)
    
    test_dataset = PatchDataset(
        train_image_files, train_mask_files,
        transform=transform_func,
        image_bands=image_bands, mask_bands=mask_bands)
    
    test_dataloader = DataLoader(
        test_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


    return test_dataloader


def parse_arguments():

    """
    parse arguments
    Returns
    -------

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action='store', type=str, help="json configuration file (required)",
                        required=True)
    parser.add_argument("-v", "--verbosity", action="store_true", help="increase output verbosity", default=0)
    args = parser.parse_args()
 
    if args.config is None or not os.path.exists(args.config):

        message = "ERROR: config file not found (check path)"
        LOGGER.error(message)
        raise OdeonError(ErrorCodes.ERR_IO, message)

    with open(args.config, 'r') as json_file:
        json_dict = JsonInterpreter(json_file)
        # json_dict.check_content(["data_sources", "model_setup"])
        return json_dict.__dict__, args.verbosity


def main():

    try:

        conf, verbosity = parse_arguments()

    except OdeonError:

        return ErrorCodes.ERR_MAIN_CONF_ERROR

    if verbosity:

        LOGGER.setLevel('DEBUG')
    else:
        LOGGER.setLevel('INFO')

    LOGGER.debug(f"Loaded configuration: \n{pformat(conf, indent=4)}")

    datasource_conf = conf.get('data_source')
    train_conf = conf.get('train_setup')
    datasource_conf.pop("val_file")
    epochs = train_conf["epochs"]
    device = train_conf["device"]    
    device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataloader = dataloader_init(
        verbosity, **datasource_conf, 
       batch_size=train_conf["batch_size"], num_workers=8)
    
    #trainer = TrainerTest(
    #    verbosity, **datasource_conf, batch_size=train_conf["batch_size"])
    #dataloader = trainer.train_dataloader

    dataloader_epochs_loop(epochs, dataloader, device)
    return 0

if __name__ == '__main__':
    
    sys.exit(main())
