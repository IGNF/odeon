import torch
from torch.nn.modules import Module


def get_output_channels_for_module(m: Module, in_channel=3, h=256, w=256):
    if m.training:
        m.eval()
    with torch.no_grad():
        input = torch.randn(2, in_channel, h, w)
        output = m(input)
        return output.shape[1]