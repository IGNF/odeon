import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DoubleConv(nn.Module):
    """Double convolution with BatchNorm as an option
    -  conv --> (BatchNorm) --> ReLu
    -  conv --> (BatchNorm) --> ReLu
        [w,h,in_ch] -> [w,h,out_ch] -> [w,h,out_ch]

    Parameters
    ----------
    in_ch : int
        number of input channels
    out_ch : int
        number of output channels
    batch_norm : bool, optional
        insert BatchNorm in double convolution, by default False
    """

    def __init__(self, in_ch, out_ch, batch_norm=False):
        super(DoubleConv, self).__init__()
        if batch_norm is True:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),  # different from original U-Net (padding is set to 0)
                nn.BatchNorm2d(out_ch),                  # original U-Net does not contain batch normalisation
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),  # different from original U-Net (padding is set to 0)
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class InputConv(nn.Module):
    """Input convolution, clone of DoubleConv
    """
    def __init__(self, in_ch, out_ch, batch_norm=False):
        super(InputConv, self).__init__()
        self.double_conv = DoubleConv(in_ch, out_ch, batch_norm)

    def forward(self, x):
        x = self.double_conv(x)
        return x


class EncoderConv(nn.Module):
    """Encoder convolution stack

    - 2x2 max-pooling with stride 2 (for downsampling)
        [w,h,in_ch] ->> [w/2,h/2,in_ch]
    - double_conv

    Parameters
    ----------
    in_ch : int
        number of input channels
    out_ch : int
        number of output channels
    batch_norm : bool, optional
        insert BatchNorm in double convolution, by default False
    """

    def __init__(self, in_ch, out_ch, batch_norm=False):
        super(EncoderConv, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, batch_norm)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class DecoderConv(nn.Module):
    """Decoder convolution stack

    - deconvolution (*2) with stride 2 upscale
        [w,h,in_ch] -> [w*2,h*2,in_ch/2]
    - concatenation
        [w*2,h*2,in_ch/2] -> [w*2,h*2,in_ch/2+in_ch/2]
    - double_conv

    Parameters
    ----------
    in_ch : int
        number of input channels
    out_ch : int
        number of output channels
    bilinear : bool, optional
        enable bilinearity in DecoderConv, by default True
    batch_norm : bool, optional
        insert BatchNorm in double convolution, by default False
    """

    def __init__(self, in_ch, out_ch, bilinear=True, batch_norm=False):
        super(DecoderConv, self).__init__()
        # upconv divide number of channels by 2 and divide widh, height by 2 with stride=2
        self.bilinear = bilinear
        if self.bilinear:
            # Conv2d reduces number of channels
            self.up = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        else:
            # ConvTranspose2d reduces number of channels and width, height
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch, batch_norm)

    def forward(self, x1, x2):
        if self.bilinear:
            # reduces width, height
            x1 = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=True)

        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x


class OutputConv(nn.Module):
    """Final layer:

    - convolution
        [w,h,in_ch] -> [w,h,out_ch]

    Parameters
    ----------
    in_ch : int
        number of input channels
    out_ch : int
        number of output channels
    """

    def __init__(self, in_ch, out_ch):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    """Convolution + Relu

    Parameters
    ----------
    in_ch : int
        number of input channels
    out_ch : int
        number of output channels
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    """DecoderBlockV2

    Parameters
    ----------
    in_ch : int
        number of input channels
    middle_ch : int
        number of input channels
    out_ch : int
        number of output channels
    is_deconv: bool, optional
        False: bilinear interpolation is used in decoder.
        True: deconvolution is used in decoder.
    """

    def __init__(self, in_ch, middle_ch, out_ch, is_deconv=True):
        super(DecoderBlockV2, self).__init__()

        if is_deconv:
            #  Paramaters for Deconvolution were chosen to avoid artifacts, following
            #  link https://distill.pub/2016/deconv-checkerboard/

            self.block = nn.Sequential(
                ConvRelu(in_ch, middle_ch),
                nn.ConvTranspose2d(middle_ch, out_ch, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvRelu(in_ch, middle_ch),
                ConvRelu(middle_ch, out_ch),
            )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net class

    Parameters
    ----------
    in_channels : int
        number of input channels
    classes : int
        number of output classes
    """

    def __init__(self, in_channels, classes):

        super(UNet, self).__init__()

        self.classes = classes

        # encoder
        self.inc = InputConv(in_channels, 64, batch_norm=True)
        self.down1 = EncoderConv(64, 128, batch_norm=True)
        self.down2 = EncoderConv(128, 256, batch_norm=True)
        self.down3 = EncoderConv(256, 512, batch_norm=True)
        self.down4 = EncoderConv(512, 1024, batch_norm=True)
        # decoder
        self.up1 = DecoderConv(1024, 512, batch_norm=True)
        self.up2 = DecoderConv(512, 256, batch_norm=True)
        self.up3 = DecoderConv(256, 128, batch_norm=True)
        self.up4 = DecoderConv(128, 64, batch_norm=True)

        self.outc = OutputConv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x


class LightUNet(nn.Module):
    """LightUnet = U-Net with reduced number of filters

    Parameters
    ----------
    in_channels : int
        number of input channels
    classes : int
        number of output classes
    """

    def __init__(self, in_channels, classes):

        super(LightUNet, self).__init__()

        self.classes = classes

        # encoder
        self.inc = InputConv(in_channels, 8)
        self.down1 = EncoderConv(8, 16)
        self.down2 = EncoderConv(16, 32)
        self.down3 = EncoderConv(32, 64)
        self.down4 = EncoderConv(64, 128)
        # decoder
        self.up1 = DecoderConv(128, 64)
        self.up2 = DecoderConv(64, 32)
        self.up3 = DecoderConv(32, 16)
        self.up4 = DecoderConv(16, 8)

        # last layer
        self.outc = OutputConv(8, classes)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x


class UNetResNet(nn.Module):

    def __init__(self, encoder_depth, classes, in_channels, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        """ U-Net model using ResNet(18, 34, 50, 101 or 152) encoder.
        UNet: https://arxiv.org/abs/1505.04597
        ResNet: https://arxiv.org/abs/1512.03385
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

        Parameters
        ----------
        encoder_depth : int
            depth of a ResNet encoder (18, 34, 50, 101 or 152).
        in_channels : int
            number of input channels
        classes : int
            number of output classes
        num_filters : int, optional
            Number of filters in the last layer of decoder, by default 32
        dropout_2d : float, optional
            probability factor of dropout layer before output layer, by default 0.2
        pretrained : bool, optional
            False: no pre-trained weights are being used.
            True: ResNet encoder is pre-trained on ImageNet,
            by default False
        is_deconv : bool, optional
            False: bilinear interpolation is used in decoder.
            True: deconvolution is used in decoder,
            by default False

        Raises
        ------
        NotImplementedError
            [description]
        """
        super().__init__()
        self.classes = classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 18:
            self.encoder = torchvision.models.resnet18(pretrained=pretrained, num_classes=classes)
            bottom_channel_nr = 512
        elif encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained, num_classes=classes)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained, num_classes=classes)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained, num_classes=classes)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained, num_classes=classes)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')

        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, classes, kernel_size=1)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))
