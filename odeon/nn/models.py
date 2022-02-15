from odeon.nn.unet import UNet, UNetResNet, LightUNet
from odeon.nn.deeplabv3p import DeeplabV3p

model_list = [
    "unet", "lightunet",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet150",
    "deeplab"
]


def build_model(model_name, n_channels, n_classes, continue_training, load_pretrained):

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
