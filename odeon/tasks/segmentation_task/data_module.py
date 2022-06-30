from pytorch_lightning import LightningDataModule


class SegmentationData(LightningDataModule):

    def __init__(self):
        super().__init__()
