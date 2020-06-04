import torch.nn as nn

import math
from collections import OrderedDict


"""
MobileNetV2

cf. https://arxiv.org/pdf/1801.04381.pdf
impl. based on https://github.com/Randl/MobileNetV2-pytorch to be able to use pretrained on imagenet weights

"""


def _make_divisible(v, divisor, min_value=None):
    """This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out

class MobileNetV2(nn.Module):
    """MobileNet2 constructor.

    Parameters
    ----------
    n_channels : int
        number of channels in the input tensor.
    n_classes : int
        number of channels in the output tensor
    width_mult : float, optional
        [description], by default 1.
    activation : :class:`Module`, optional
        activation function, by default nn.ReLU6
    """
    def __init__(self, n_channels, n_classes, width_mult=1., activation=nn.ReLU6):

        super(MobileNetV2, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # sample size (width, height) must be multiple of 32
        # assert input_size % 32 == 0
        # input_channel: input of bottleneck blocks must be divisable by 8
        input_channel = _make_divisible(32 * width_mult, 8)

        # first layer is Conv3x3 layer
        self.conv1 = nn.Conv2d(self.n_channels, input_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.activation = activation(inplace=True)

        # inverted residual blocks (bottleneck)
        cfgs = [
            # t (expansion factor), c (channels), n (repeated n times), s (stride)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.bottlenecks_modules = OrderedDict()
        i_cfg = 0
        for t, c, n, s in cfgs:
            stage_modules = OrderedDict()
            output_channel = _make_divisible(c * width_mult, 8)
            stage_modules[f"LinearBottleneck{i_cfg}_0"] = LinearBottleneck(input_channel, output_channel, s, t)
            input_channel = output_channel
            # handle number of repeat times (n)
            for i in range(1, n):
                stage_modules[f"LinearBottleneck{i_cfg}_{i}"] = LinearBottleneck(input_channel, output_channel, 1, t)
                input_channel = output_channel

            self.bottlenecks_modules[f"Bottlenecks_{i_cfg}"] = nn.Sequential(stage_modules)
            i_cfg += 1

        self.bottlenecks = nn.Sequential(self.bottlenecks_modules)

        # last several layers (not being used in deeplab)
        # self.last_conv_out_ch = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        # self.conv_last = nn.Conv2d(input_channel, self.last_conv_out_ch, kernel_size=1, bias=False)
        # self.bn_last = nn.BatchNorm2d(self.last_conv_out_ch)
        # self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        # self.dropout = nn.Dropout(0.2)
        # self.fc = nn.Linear(self.last_conv_out_ch, n_classes)

        self._initialize_weights()

    def forward(self, x):

        # low level features are first conv layer and 2 first bottleneck layers

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.bottlenecks_modules["Bottlenecks_0"](x)
        x = self.bottlenecks_modules["Bottlenecks_1"](x)

        low_level_feat = x

        for i in range(2, 7):
            x = self.bottlenecks_modules[f"Bottlenecks_{i}"](x)

        return x, low_level_feat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
