from torch import nn
from torchvision.models import MobileNetV2

try:
    from torchvision.models.mobilenetv2 import _make_divisible, ConvBNReLU
except ModuleNotFoundError:
    from torchvision.models.mobilenet import _make_divisible, ConvBNReLU


class MobileNetV2(MobileNetV2):
    """MobileNetV2 version used as deeplab backbone
    First layer is rewritten to accept a number of channels != 3.
    Low features are extracted to be reinjected in deeplab decoder

    Parameters
    ----------
    n_channels : int
        number of input image channel
    n_classes : int
        number of output classes
    width_mult : float, optional
        cf. <torchvision.models.MobileNetV2>, by default 1.0
    inverted_residual_setting : [type], optional
        cf. <torchvision.models.MobileNetV2>, by default None
    round_nearest : int, optional
        cf. <torchvision.models.MobileNetV2>, by default 8
    block : [type], optional
        cf. <torchvision.models.MobileNetV2>, by default None
    """
    def __init__(self,
                 n_channels,
                 n_classes,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        super(MobileNetV2, self).__init__()

        # replace original first layer which is made for 3-channels images
        input_channel = 32
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.features[0] = ConvBNReLU(n_channels, input_channel, stride=2)

        # extract low features
        self.low_features = nn.Sequential(self.features[0:3])

        # remove last layer
        self.features = self.features[:-1]

    def forward(self, x):
        low_feat = self.low_features(x)

        # x = super(MobileNetV2, self).forward(x)
        x = self.features(x)

        return x, low_feat
