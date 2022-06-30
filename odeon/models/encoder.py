import timm
import torch.nn as nn
encoder_names = ["resnet18", "resnet34"]


class Encoder_Factory:

    def __init__(self,
                 encoder_name="resnet18",
                 pretrained=True,
                 in_chans=3,
                 checkpoint=None,
                 features_only=True):

        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.checkpoint =checkpoint
        self.features_only = features_only

    def get(self):

        if self.checkpoint is not None:

            encoder: nn.Module = timm.create_model(self.encoder_name,
                                                   pretrained=False,
                                                   in_chans=self.in_chans,
                                                   checkpoint_path=self.checkpoint)

        else:

            encoder: nn.Module = timm.create_model(self.encoder_name,
                                                   pretrained=self.pretrained,
                                                   in_chans=self.in_chans)

        return encoder


