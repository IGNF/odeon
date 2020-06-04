from odeon.nn.unet import UNet, UNetResNet, HeavyUNet
from odeon.nn.mobilenetv2 import MobileNetV2
from torchvision.models.segmentation import DeepLabV3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

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
        backbone = MobileNetV2(n_channels, n_classes)
        classifier = DeepLabHead(n_channels, n_classes)
        net = DeepLabV3(backbone, classifier)

        # net = deeplab.DeeplabV3p(n_channels=n_channels, n_classes=n_classes,
        #                          input_size=cfg['data_loader']['image_setup']['side'], device=device,
        #                          load_pretrained=load_pretrained)

    return net
