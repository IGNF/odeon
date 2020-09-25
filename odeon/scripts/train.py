"""Training

This module trains a semantic segmentation model from sample files.



Example
-------
    Call this module from the root of the project:

    $ odeon train -c src/json/train.json -v

    This will read the configuration from a json file and train a model.
    Model is stored in output_folder in a .pth file.


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

from odeon.nn.transforms import Compose, Rotation90, Rotation, Radiometry, ToDoubleTensor
from odeon.nn.datasets import PatchDataset
from odeon.nn.training_engine import TrainingEngine
from odeon.nn.models import build_model
from odeon.nn.losses import CrossEntropyWithLogitsLoss, FocalLoss2d, ComboLoss

logger = logging.getLogger(__package__)


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
            return CrossEntropyWithLogitsLoss(weight=weight)

        else:

            return CrossEntropyWithLogitsLoss()

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

    dataset : :class:`Dataset`

    Returns
    -------
    tuple
        width, height, n_bands
    """

    sample = dataset.__getitem__(0)

    return {'image': sample['image'].shape, 'mask': sample['mask'].shape}


def train(verbose, train_file, model_name, output_folder, val_file=None, percentage_val=0.2, image_bands=None,
          mask_bands=None, model_filename=None, load_pretrained_enc=False, epochs=300, batch_size=16, patience=20,
          save_history=False, continue_training=False, loss="ce", class_imbalance=None, optimizer="adam", lr=0.001,
          data_augmentation=None, device=None, reproducible=False):
    """[summary]


    Parameters
    ----------
    verbose : bool
        verbose level
    train_file : str
        CSV file with image files in this first column and mask files in the second
    model_name : str
        name of model within ('unet', 'deeplab')
    output_folder : str
        path to output folder
    val_file : str, optional
        CSV file for validation, by default None
    percentage_val : number, optional
        used if val_file is None, by default 0.2
    image_bands : list of int, optional
        list of band indices, by default None
    mask_bands : list of int, optional
        list of band indices, by default None
    model_filename : str, optional
        name of pth file, if None model name will be used, by default None
    load_pretrained_enc : bool, optional
        WIP: load pretrained weights for encoder, by default False
    epochs : int, optional
        number of epochs, by default 300
    batch_size : int, optional
        batch size, by default 16
    patience : int, optional
        maximum number of epoch without improvement before train is stopped, by default 20
    save_history : bool, optional
        activate history storing, by default False
    continue_training : bool, optional
        resume a training, by default False
    loss : str, optional
        loss function within ('ce', 'bce', 'wce', 'focal', 'combo'), by default "ce"
    class_imbalance : list of number, optional
        weights for weighted-cross entropy loss, by default None
    optimizer : str, optional
        optimizer name within ('adam', 'SGD'), by default "adam"
    lr : number, optional
        start learning rate, by default 0.001
    data_augmentation : list, optional
        list of data augmentation function within ('rotation', 'rotation90', 'radiometry'), by default None
    device : str, optional
        device if None 'cpu' or 'cuda' if available will be used, by default None
    reproducible : bool, optional
        activate training reproducibility, by default False
    """

    # reproducibility
    if reproducible is True:
        random_seed = 2020
    else:
        random_seed = None

    # transformations
    if data_augmentation is None:
        data_augmentation = ['rotation90']
    transformation_dict = {
        "rotation90": Rotation90(),
        "rotation": Rotation(),
        "radiometry": Radiometry()
    }
    transformation_conf = data_augmentation
    transformation_keys = transformation_conf if isinstance(transformation_conf, list) else [transformation_conf]

    transformation_functions = list({
        value for key, value in transformation_dict.items() if key in transformation_keys
    })
    transformation_functions.append(ToDoubleTensor())
    logger.info(f"Data augmentation: {transformation_keys}")

    # datasets & dataloaders
    #   read csv file with columns: image, mask
    train_image_files, train_mask_files = read_csv_sample_file(train_file)
    if val_file:

        val_image_files, val_mask_files = read_csv_sample_file(val_file)

    else:

        train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(
            train_image_files, train_mask_files, test_size=percentage_val, random_state=random_seed)

    logger.info(
        f"Selection of {len(train_image_files)} files for training and {len(val_image_files)} for model validation")

    assert batch_size <= len(train_image_files), "batch_size must be lower than the length of training dataset"
    train_dataset = PatchDataset(train_image_files, train_mask_files, transform=Compose(transformation_functions),
                                 image_bands=image_bands, mask_bands=mask_bands)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8)
    val_dataset = PatchDataset(val_image_files, val_mask_files, transform=Compose(transformation_functions),
                               image_bands=image_bands, mask_bands=mask_bands)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=8)

    # training
    #    model generation
    if image_bands is not None:
        n_channels = len(image_bands)
    else:
        n_channels = get_sample_shape(train_dataset)['image'][0]
    if mask_bands is not None:
        n_classes = mask_bands
    else:
        n_classes = get_sample_shape(train_dataset)['mask'][0]
    model = build_model(model_name, n_channels, n_classes)
    #    optimizer
    optimizer_function = get_optimizer(optimizer, model, lr)
    #    loss
    loss_function = get_loss(loss, class_weight=class_imbalance)
    #    learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer_function, 'min', factor=0.5, patience=10, verbose=verbose,
                                     cooldown=4, min_lr=1e-7)

    #    training engine instanciation
    if model_filename is None:
        model_filename = f"{model_name}.pth"
    training_engine = TrainingEngine(model, loss_function, optimizer_function, lr_scheduler, output_folder,
                                     model_filename, epochs=epochs, batch_size=batch_size, patience=patience,
                                     save_history=save_history, continue_training=continue_training,
                                     reproducible=reproducible, device=device, verbose=verbose)

    net_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters trainable : {net_params}")

    try:

        training_engine.train(train_dataloader, val_dataloader)

    except KeyboardInterrupt:
        tmp_file = os.path.join('/tmp', 'INTERRUPTED.pth')
        tmp_optimizer_file = os.path.join('/tmp', 'optimizer_INTERRUPTED.pth')
        torch.save(model.state_dict(), tmp_file)
        torch.save(optimizer_function.state_dict(), tmp_optimizer_file)
        logger.info(f"Saved interrupt as {tmp_file}")
