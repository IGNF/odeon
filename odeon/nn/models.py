import os
import torch
from pathlib import Path
from odeon.nn.unet import UNet, UNetResNet, LightUNet
from odeon.nn.deeplabv3p import DeeplabV3p

model_list = [
    "unet", "lightunet",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet150",
    "deeplab"
]


def build_model(model_name,
                n_channels, 
                n_classes,                                  
                init_model_weights=None,
                load_pretrained_weights=None):

    if model_name == 'lightunet':
        net = LightUNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'unet':
        net = UNet(n_channels=n_channels, n_classes=n_classes)
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
