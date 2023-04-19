"""
"""
from pytorch_lightning import LightningDataModule

from odeon.core.logger import get_logger

LOGGER = get_logger(__name__)


class OdnDataModule(LightningDataModule):
    def __call__(self, *args, **kwargs):
        LOGGER.debug(f'{self}')
