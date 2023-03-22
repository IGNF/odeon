""" Fit command line module """
from dataclasses import asdict

from jsonargparse import ArgumentParser, Namespace, set_config_read_mode

from odeon import ENV, LOGGER
from odeon.fit.app import (FitConfig, InputConfig, ModelConfig, SeedConfig,
                           StageConfig, TrainerConfig)

# from typing import Dict, Optional
# from odeon.data.data_module import Input
# from odeon.models.py.change.module.change_unet import ChangeUnet

URL_ENABLED = True
set_config_read_mode(urls_enabled=URL_ENABLED)  # enable URL as path file
PATH = '__path__'


def main(cfg: Namespace, parser: ArgumentParser) -> None:
    instantiated_config = parser.instantiate_classes(cfg=cfg)
    LOGGER.debug(f'config: {cfg}')
    LOGGER.debug(f'instantiated config: {instantiated_config}')


if __name__ == '__main__':
    parser = ArgumentParser(parser_mode=ENV.config.config_parser)
    parser.add_dataclass_arguments(nested_key='model', theclass=ModelConfig, default=None)
    parser.add_dataclass_arguments(nested_key='data', theclass=InputConfig, default=None)
    parser.add_dataclass_arguments(nested_key='trainer', theclass=TrainerConfig, default=None)
    parser.add_dataclass_arguments(nested_key='stage', theclass=StageConfig, default=None)
    parser.add_dataclass_arguments(nested_key='seed', theclass=SeedConfig, default=None)
    parser.add_argument('--conf', type=FitConfig, default=None, help=FitConfig.__doc__)
    cfg = parser.parse_args()
    LOGGER.debug(cfg)
    LOGGER.info(asdict(ENV.config))
    main(cfg=cfg, parser=parser)
