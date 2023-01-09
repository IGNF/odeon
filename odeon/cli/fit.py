from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

# from odeon.data.data_module import Input
# from odeon.models.py.change.module.change_unet import ChangeUnet

# TODO : change help message


class FitParser(LightningArgumentParser):
    def __init__(self):
        super(FitParser, self).__init__()


class FitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        ...


def main():
    ...
    # cli = FitCLI(model_class=ChangeUnet, datamodule_class=Input)
