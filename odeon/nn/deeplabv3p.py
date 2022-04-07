import torch
from torch import nn
from torch.nn import functional as F
from odeon.nn.mobilenetv2 import MobileNetV2
from torchvision.models.segmentation.deeplabv3 import ASPP


class Decoder(nn.Module):
    """Decoder module for deeplab model

    Parameters
    ----------
    num_classes : int
        number of output classes
    backbone : str
        name of backbone used in deeplab within ('resnet', 'xception', 'MobileNetV2')

    Raises
    ------
    NotImplementedError
        [description]
    """
    def __init__(self, num_classes, backbone):

        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'MobileNetV2':
            low_level_inplanes = 24
        else:

            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x


class DeeplabV3p(nn.Module):
    """Deeplab V3 + implementation
    cf. https://github.com/tensorflow/models/tree/master/research/deeplab

    Parameters
    ----------
    n_channels : int
        number of channels in input image
    n_classes : int
        number of output classes
    output_stride : int, optional
        output stride, by default 8
    """

    def __init__(self, in_channels, classes, output_stride=8):
        super(DeeplabV3p, self).__init__()
        self.n_classes = classes
        self.backbone = MobileNetV2(n_classes=classes, n_channels=in_channels)
        if output_stride == 16:
            dilatations = [6, 12, 18]
        elif output_stride == 8:
            dilatations = [12, 24, 36]
        self.aspp = ASPP(320, dilatations)
        self.decoder = Decoder(classes, type(self.backbone).__name__)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x
