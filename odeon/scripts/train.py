"""Training

This module trains a semantic segmentation model from sample files.



Example
-------
    Call this module from the root of the project:

    $ python odeon/main.py train -c src/json/train.json -v

    This will read the configuration from a json file and train a model.
    Model is stored in output_path folder in a .pth file.


Notes
-----


"""

import os
import csv
import logging
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from odeon.commons.timer import Timer

from odeon.nn.transforms import Compose, Rotation90, Rotation, Radiometry, ToDoubleTensor
from odeon.nn.datasets import PatchDataset
from odeon.nn.training_engine import TrainingEngine
from odeon.nn.models import build_model
from odeon.nn.losses import FocalLoss2d, ComboLoss

logger = logging.getLogger(__package__)

def read_csv_sample_file(file_path):
    image_files = []
    mask_files = []
    if not os.path.exists(file_path):
        logger.error(f"{file_path} does not exists.")
    with open(file_path) as csvfile:
        sample_reader = csv.reader(csvfile)
        for item in sample_reader:
            image_files.append(item[0])
            mask_files.append(item[1])
    return image_files, mask_files


def get_optimizer(optimizer_name, model, lr):
    """Initialize optimizer object from name

    Parameters
    ----------
    optimizer_name : str
        optimizer name possible values = ("adam", "SGD")
    model : nn.Module
        pytorch neural network object
    lr : float
        learning rate
    Returns
    -------
        torch.Optimizer
    """

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr)

def get_loss(loss_name, class_weight=None, use_cuda=False):
    """Initialize loss class instance
       Loss function applied directly on models raw prediction (logits)

    Parameters
    ----------
    loss_name : str
        loss name, possible values = ("ce", "bce", "focal", "combo")
    class_weight : list of float, optional
        weights applied to each class in loss computation, by default None
    use_cuda : bool, optional
        use CUDA, by default False

    Returns
    -------
    [type]
        [description]
    """

    if loss_name == "ce":
        if class_weight is not None:
            logger.info(f"Weights used: {class_weight}")
            weight = torch.FloatTensor(class_weight)
            if use_cuda:
                weight = weight.cuda()
            return nn.CrossEntropyLoss(weight=weight)
        else:
            return nn.CrossEntropyLoss()
    elif loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "focal":
        return FocalLoss2d()
    elif loss_name == "combo":
        return ComboLoss({'bce': 0.75, 'jaccard': 0.25})

def get_sample_shape(dataset):
    """get sample shape from dataloader

    Parameters
    ----------
    dataloader : :class:`DataLoader`

    Returns
    -------
    tuple
        width, height, n_bands
    """

    sample = dataset.__getitem__(0)

    return {'image': sample['image'].shape, 'mask': sample['mask'].shape}


def train(conf):
    with Timer("Training"):

        # reproducibility
        if conf.get('train_setup', {}).get('reproducible', False):
            random_seed = 2020
        else:
            random_seed = None

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
        transformation_functions.append(ToDoubleTensor())
        logger.info(f"Data augmentation: {transformation_keys}")

        # datasets & dataloaders
        #   read csv file with columns: image, mask
        train_csv_file = conf.get('data_sources').get('train', None)
        train_image_files, train_mask_files = read_csv_sample_file(train_csv_file)
        val_csv_file = conf.get('data_sources').get('val', None)
        if val_csv_file:
            val_image_files, val_mask_files = read_csv_sample_file(val_csv_file)
        else:
            percentage_val = conf.get('data_sources').get('percentage_val', 0.2)
            train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(
                train_image_files, train_mask_files, test_size=percentage_val, random_state=random_seed)

        logger.info(
            f"Selection of {len(train_image_files)} files for training and {len(val_image_files)} for model validation")

        #    get parameters for dataset initialisation
        dataset_optional_args = {}
        for key in ['width', 'height', 'image_bands', 'mask_bands']:
            dataset_optional_args[key] = conf.get('train_setup', {}).get(key, None)

        train_dataset = PatchDataset(train_image_files, train_mask_files,
                                     transform=Compose(transformation_functions), **dataset_optional_args)
        train_dataloader = DataLoader(train_dataset)
        val_dataset = PatchDataset(val_image_files, val_mask_files,
                                   transform=Compose(transformation_functions), **dataset_optional_args)
        val_dataloader = DataLoader(val_dataset)

        # training
        epochs = conf.get('train_setup', {}).get('epochs', 300)
        training_optional_args = conf.get('train_setup', {})

        #    model generation
        if dataset_optional_args['image_bands'] is not None:
            n_channels = len(dataset_optional_args['image_bands'])
        else:
            n_channels = get_sample_shape(train_dataset)['image'][0]
        if dataset_optional_args['mask_bands'] is not None:
            n_classes = len(dataset_optional_args['mask_bands'])
        else:
            n_classes = get_sample_shape(train_dataset)['mask'][0]
        model_name = conf.get('model_setup').get('name')
        model = build_model(model_name, n_channels, n_classes)
        #    optimizer
        optimizer_name = conf.get('model_setup').get('optimizer', 'adam')
        lr = conf.get('model_setup').get('lr', 0.001)
        optimizer = get_optimizer(optimizer_name, model, lr)
        #    loss
        loss_name = conf.get('model_setup').get('loss', 'ce')
        loss_function = get_loss(loss_name)
        #    learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True, cooldown=4,
                                         min_lr=1e-7)

        #    training engine instanciation
        output_folder = conf.get('model_setup').get('output_folder')
        training_engine = TrainingEngine(model, loss_function, optimizer, lr_scheduler, output_folder, epochs=epochs,
                                         **training_optional_args)

        net_params = sum(p.numel() for p in model.parameters())
        net_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters (trainable/total) : {net_params} / {net_total_params}")

        try:
            training_engine.train(train_dataloader, val_dataloader)

        except KeyboardInterrupt:
            tmp_file = os.path.join('/tmp', 'INTERRUPTED.pth')
            torch.save(model.state_dict(), tmp_file)
            logger.info(f"Saved interrupt as {tmp_file}")
