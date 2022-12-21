import torch

from odeon.models.change.arch.change_unet import FCSiamConc, FCSiamDiff


def test_change_unet():

    image_batch = torch.rand(8, 2, 5, 512, 512)
    fc_siam_conc_model = FCSiamConc(in_channels=5)
    out = fc_siam_conc_model(image_batch)
    assert out.shape == (8, 1, 512, 512)
