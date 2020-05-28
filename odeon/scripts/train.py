"""Training

This module trains a semantic segmentation model from sample files.



Example
-------
    Call this module from the root of the project:

    $ python -m src.train -c src/json/train.json -v

    This will read the configuration from a json file and train a model. Model is stored in output_path folder in a .pth file.


Notes
-----
    * [Todo] implement default values for "image_size_pixel" and "pixel_size_meter_per_pixel" so they can be
    skipped in json (see json_interpreter)

"""

import os
import csv
import logging
from sys import exit
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchvision import transforms

from odeon.commons.timer import Timer

from odeon.nn.transforms import Rotation90, Rotation, Radiometry
from odeon.nn.datasets import PatchDataset
from odeon.nn.training_engine import TrainingEngine


def read_csv_sample_file(file_path):
    image_files = []
    mask_files = []
    if not os.path.exists(file_path):
        logging.error(f"{file_path} does not exists.")
    with open(file_path) as csvfile:
        sample_reader = csv.reader(csvfile)
        for item in sample_reader:
            image_files.append(item[0])
            mask_files.append(item[1])
    return image_files, mask_files

def train(conf):
    with Timer("Training"):

        # reproducibility
        if conf.get('train_setup', {}).get('reproducible', False):
            random_seed = 2020
        else:
            random_seed = None

        # datasets & dataloaders
        # read csv file with columns: image, mask
        train_csv_file = conf.get('data_sources').get('train', None)
        train_image_files, train_mask_files = read_csv_sample_file(train_csv_file)
        val_csv_file = conf.get('data_sources').get('val', None)
        if val_csv_file:
            val_image_files, val_mask_files = read_csv_sample_file(val_csv_file)
        else:
            percentage_val = conf.get('data_sources').get('percentage_val', 0.2)
            train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(
                train_image_files, train_mask_files, test_size=percentage_val, random_state=random_seed)

        logging.info(
            f"Selection of {len(train_image_files)} files for training and {len(val_image_files)} for model validation")

        # transformations
        transformation_dict = {
            "rotation90": Rotation90(),
            "rotation": Rotation(),
            "radiometry": Radiometry()
        }
        transformation_keys = ["rotation90"]
        for trans in conf.get('train_setup', {}).get('data_augmentation', []):
            transformation_keys.append(trans)
        transformation_functions = list({
            value for key, value in transformation_dict.items() if key in transformation_keys
        })
        transformations = transforms.Compose(transformation_functions)
        logging.info(f"Data augmentation: {transformation_keys}")

        dataset_optional_args = {}
        for key in ['width', 'height', 'image_bands', 'mask_bands']:
            dataset_optional_args[key] = conf.get('train_setup', {}).get(key, None)
        train_dataset = PatchDataset(train_image_files, train_mask_files, transformations=transformations,
                                     **dataset_optional_args)
        train_dataloader = DataLoader(train_dataset)
        val_dataset = PatchDataset(val_image_files, val_mask_files, transformations=transformations,
                                   **dataset_optional_args)
        val_dataloader = DataLoader(val_dataset)

        # if conf.get('train_setup', {}).get('epochs', None):
        #     optional_args['epochs'] = conf.get('train_setup').get('epochs')
        # if conf.get('train_setup', {}).get('batch_size', None):
        #     optional_args['batch_size'] = conf.get('train_setup').get('batch_size')

        training_optional_args = conf.get('train_setup', {})

        model_name = conf.get('model_setup').get('model_name')
        training_engine = TrainingEngine(model_name, )

        training_engine.train(train_dataloader, val_dataloader)
