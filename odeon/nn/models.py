from odeon.nn.unet import UNet, UNetResNet, HeavyUNet
from odeon.nn.deeplabv3p import DeeplabV3p

model_list = ["unet", "heavyunet", "resnet", "deeplab"]


def build_model(model_name, n_channels, n_classes, load_pretrained=False):
    """Build a nn model from a model name.

    Parameters
    ----------
    model_name : str
        model name, possible values:
        'unet', 'heavyunet',
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

    if model_name == 'unet':
        net = UNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'heavyunet':
        net = HeavyUNet(n_channels=n_channels, n_classes=n_classes)
    elif str.startswith(model_name, 'resnet'):
        depth = int(model_name[6:])
        net = UNetResNet(depth, n_classes=n_classes, n_channels=n_channels)
    elif model_name == 'deeplab':
        net = DeeplabV3p(n_channels, n_classes, output_stride=16)

    return net
