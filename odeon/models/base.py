import os
import torch
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.models import (
    UNet,
    UNetResNet, 
    LightUNet, 
    DeeplabV3p
)

model_list = [
    "unet", "lightunet",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet150",
    "deeplab"
]

def build_model(model_name,
                n_channels, 
                n_classes,
                deterministic=False):

    bilinear = False if deterministic else True
    if deterministic and model_name not in ["unet", "lightunet"]:
        LOGGER.error('ERROR: The reproductibility of a training only works for Unet and Lightunet models in this version.')
        raise OdeonError(ErrorCodes.ERR_MODEL_ERROR,
            "Wrong parameters model_name or reproducible. If reproducible is True model name should be 'unet' or 'lightunet'")

    if model_name == 'lightunet':
        net = LightUNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    elif model_name == 'unet':
        net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    elif str.startswith(model_name, 'resnet'):
        depth = int(model_name[6:])
        net = UNetResNet(depth, n_classes=n_classes, n_channels=n_channels)
    elif model_name == 'deeplab':
        net = DeeplabV3p(n_channels, n_classes, output_stride=16)
    return net


def get_train_filenames(out_dir, out_filename):
    model_file = os.path.join(out_dir, out_filename)
    optimizer_filename = f'optimizer_{out_filename}'
    optimizer_file = os.path.join(out_dir, optimizer_filename)
    train_filename = f'train_{out_filename}'
    train_file = os.path.join(out_dir, train_filename)
    base_history_path = os.path.join(out_dir, f'{os.path.splitext(out_filename)[0]}')
    history_file = f'{base_history_path}_history.json'
    return {
        "model": model_file,
        "optimizer": optimizer_file,
        "train": train_file,
        "base": base_history_path,
        "history": history_file}


def save_model(out_dir, out_filename, model, optimizer=None, scheduler=None, train_dict=None):
    train_filenames = get_train_filenames(out_dir, out_filename)

    torch.save(model.state_dict(), train_filenames["model"])
    if optimizer is not None:
        torch.save(optimizer.state_dict(), train_filenames["optimizer"])
    if train_dict is not None or scheduler is not None:
        save_train_dict = train_dict if train_dict is not None else dict()
        if scheduler is not None:
            save_train_dict["scheduler"] = scheduler.state_dict()
        torch.save(save_train_dict, train_filenames["train"])
    return train_filenames["model"]
