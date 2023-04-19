""" Fit command line module """
from dataclasses import asdict

from jsonargparse import ArgumentParser, Namespace, set_config_read_mode

from odeon import ENV
# from odeon.core.exceptions import MisconfigurationException
from odeon.core.logger import get_logger
from odeon.fit.app import FitApp, FitConfig

# from typing import Optional


LOGGER = get_logger(logger_name=__name__, debug=ENV.config.debug_mode)
# from typing import Dict, Optional
# from odeon.data.data_module import Input
# from odeon.models.py.change.module.change_unet import ChangeUnet

URL_ENABLED = True
set_config_read_mode(urls_enabled=URL_ENABLED)  # enable URL as path file
PATH = '__path__'


def main(cfg: Namespace, parser: ArgumentParser) -> None:

    # LOGGER.debug(f'config: {cfg}')
    """
    try:
        assert (cfg.data is not None) or (cfg.conf is not None)
    except AssertionError as e:
        raise MisconfigurationException(message=f"you need to fill one of those parameters:"
                                                f"--conf or --data")
    """
    """
    if cfg.conf is None:
        instanciated_cfg = parser.instantiate_classes(cfg=cfg)
        fit_config: FitConfig = FitConfig(input_config=instanciated_cfg.data,
                                          model_config=instanciated_cfg.model,
                                          trainer_config=instanciated_cfg.trainer,
                                          stage_config=instanciated_cfg.stage,
                                          seed_config=instanciated_cfg.seed)
    else:
        cfg.conf.data = cfg.data
        cfg.conf.model = cfg.model
        cfg.conf.trainer = cfg.trainer
        cfg.conf.seed = cfg.seed
        cfg.conf.stage = cfg.stage
    """
    instanciated_cfg = parser.instantiate_classes(cfg=cfg)
    fit_config: FitConfig = instanciated_cfg.conf
    fit_app = FitApp(config=fit_config)
    fit_app()
    # LOGGER.debug(fit_app.__dict__)
    LOGGER.debug(f'fit app: {fit_app}')


if __name__ == '__main__':
    parser = ArgumentParser(parser_mode=ENV.config.config_parser)
    """
    parser.add_argument('--model', type=ModelConfig, default=None, help=ModelConfig.__doc__)
    parser.add_argument('--data', type=InputConfig, default=None, help=InputConfig.__doc__)
    parser.add_argument('--trainer', type=Optional[TrainerConfig], default=None, help=TrainerConfig.__doc__)
    parser.add_argument('--stage', type=Optional[StageConfig], default=None, help=StageConfig.__doc__)
    parser.add_argument('--seed', type=Optional[SeedConfig], default=None, help=FitConfig.__doc__)

    parser.add_argument('--conf', type=Optional[FitConfig], default=None, help=FitConfig.__doc__)

    parser.add_dataclass_arguments(nested_key='model', theclass=ModelConfig, default=None)
    parser.add_dataclass_arguments(nested_key='data', theclass=InputConfig, default=None)
    parser.add_dataclass_arguments(nested_key='trainer', theclass=TrainerConfig, default=None)
    parser.add_dataclass_arguments(nested_key='stage', theclass=StageConfig, default=None)
    parser.add_dataclass_arguments(nested_key='seed', theclass=SeedConfig, default=None)
    """
    parser.add_dataclass_arguments(nested_key='conf', theclass=FitConfig)
    # parser.link_arguments('model', 'conf.model_config', apply_on='parse')
    cfg = parser.parse_args()
    LOGGER.debug(cfg)
    LOGGER.debug(f'env configuration: {asdict(ENV.config)}')
    main(cfg=cfg, parser=parser)
