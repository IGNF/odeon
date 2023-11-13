import os
from pathlib import Path
import torch
from odeon.nn.unet import UNet, UNetResNet, LightUNet
from odeon.nn.deeplabv3p import DeeplabV3p
from odeon.commons.exception import OdeonError, ErrorCodes

model_list = [
    "unet", "lightunet",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet150",
    "deeplab"
]


def build_model(model_name, n_channels, n_classes, load_pretrained=False):
    """Build a nn model from a model name.

    Parameters
    ----------
    model_name : str
        model name, possible values:
        'lightunet', 'unet',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet150',
        'deeplab'
    n_channels : int
        number of channels in the input image
    n_classes : int
        number of classes in the output mask
    load_pretrained : bool, optional
        load pretrained weights for model, by default False

    Returns
    -------
    :class:`nn.Module`
        pytorch neural network model
    """

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
    """
    return dict of path used to save training info

    return a dictionnary with the path of the model file, optimize file, history_file
    which are all needed to resume a training with the option continue_training

    Parameters
    ----------
    out_dir : str
        Path where models files will be saved.
    out_filename : str
        name of the basename for the model file from which the other basename
        will be derived

    Returns
    -------
    :dict:
        dictionnary with absolute path as value and file type as key

    Examples
    --------
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> train_dict = get_train_filenames("/home/adupont/test_odeon", "unet_ocs_bce.pth")
    >>> train_dict["model"]
    "/home/adupont/test_odeon/unet_ocs_bce.pth"
    >>> train_dict
    {"model": "/home/adupont/test_odeon/unet_ocs_bce.pth",
    "optimizer": "/home/adupont/test_odeon/optimizer_unet_ocs_bce.pth",
    "train": "/home/adupont/test_odeon/train_unet_ocs_bce.pth",
    "base": "/home/adupont/test_odeon/unet_ocs_bce",
    "history": "/home/adupont/test_odeon/unet_ocs_bce_history.json"
    }

    """
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


def load_model(model_name, model_path, n_channel, n_classes, use_gpu=False):
    """load model from a model name and models files

    Parameters
    ----------
    model_name : str
        model name, possible values:
        'lightunet', 'unet',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet150',
        'deeplab'
    model_path : str
        full filename (path) of a save model. generally a *.pth file.
    n_channels : int
        number of channels in the input image
    n_classes : int
        number of classes in the output mask
    use_gpu : bool, optional
        load the model to gpu (training mode) or not (detect on cpu)

    Returns
    -------
    :class:`nn.Module`
        pytorch neural network model

    """

    if model_name not in model_list:

        raise OdeonError(message=f"the model name {model_name} does not exist",
                         error_code=ErrorCodes.ERR_MODEL_ERROR)

    model = build_model(model_name, n_channel, n_classes)

    if use_gpu:

        model.cuda()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict=state_dict)

    else:

        model.cpu()
        state_dict = torch.load(model_path,
                                map_location=torch.device('cpu'))
        # LOGGER.debug(state_dict.keys())
        model.load_state_dict(state_dict=state_dict)

    return model


def save_model(out_dir, out_filename, model, optimizer=None, scheduler=None, train_dict=None):
    """
    save a model and optionaly it's training info

    Parameters
    ----------
    out_dir : str
        Path of directory where models files will be saved.
    out_filename : str
        name of the basename for the model file from which the other basename
        will be derived
    model : nn.Module
        odeon/pytorch model to save
    optimizer : nn.Module
        optimizer used for training. If not None it state are saved in optimizer_*.pth
        to enable continue_training and resume the training state.
    scheduler : nn.Module
        scheduler used for training. If not None it state are saved in train_*.pth
        into scheduler key
    train_dict : dict
        optional training info to save in train_*.pth file.

    Returns
    -------
    :str:
        path to the saved model
    """
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


def resume_train_state(out_dir, out_filename, optimizer=None, scheduler=None):
    """
    resume training optimizer, scheduler to last training state.

    Parameters
    ----------
    out_dir : str
        Path of directory where models files will be saved.
    out_filename : str
        name of the basename for the model file from which the other basename
        will be derived
    optimizer : nn.Module
        optimizer used for training.
    scheduler : nn.Module
        scheduler used for training.

    """
    train_filenames = get_train_filenames(out_dir, out_filename)
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(train_filenames["optimizer"]))
    if scheduler is not None and Path(train_filenames["train"]).exists():
        train_dict = torch.load(train_filenames["train"])
        scheduler.load_state_dict(train_dict["scheduler"])
