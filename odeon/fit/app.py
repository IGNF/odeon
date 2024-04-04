from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, cast

from pytorch_lightning import (LightningDataModule, LightningModule,
                               seed_everything)
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# from odeon introspection.py LOGGER
from odeon.core.app import App
from odeon.core.exceptions import MisconfigurationException
from odeon.core.singleton import Singleton
from odeon.core.types import PARAMS, STAGES_OR_VALUE, Stages
from odeon.data.core.registry import DATA_REGISTRY
from odeon.models.core.models import MODEL_REGISTRY

from .callbacks import build_callbacks
from .core.types import OdnCallback, OdnLogger
from .logger import build_loggers
from .trainer import OdnTrainer

STAGE_ORDER = {str(Stages.FIT.value): 1,
               Stages.FIT: 2,
               str(Stages.VALIDATE.value): 3,
               Stages.VALIDATE: 4,
               str(Stages.TEST.value): 5,
               Stages.TEST: 6,
               str(Stages.PREDICT.value): 7,
               Stages.PREDICT: 8}
FIT_STAGES: List[Stages | str] = [str(Stages.FIT.value)]
INFERENCE_STAGES: List[Stages | str] = [str(Stages.VALIDATE.value),
                                        Stages.VALIDATE,
                                        str(Stages.TEST.value),
                                        Stages.TEST,
                                        str(Stages.PREDICT.value),
                                        Stages.PREDICT]
CKPT_PATH = 'ckpt_path'
DEFAULT_CKPT_PATH_INFERENCE: str = 'best'
DEFAULT_INFERENCE_PARAMS: PARAMS = {CKPT_PATH: DEFAULT_CKPT_PATH_INFERENCE}
# TRAINER_PARAM_FIELD: str = 'trainer_params'
DEFAULT_MODEL_NAME = 'change_unet'
DEFAULT_INPUT_NAME = 'input'


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class InputConfig:

    input_name: str = 'input'
    input_params: PARAMS = field(default_factory=lambda: dict())


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class ModelConfig:
    model_name: str = 'change_unet'
    model_params: PARAMS = field(default_factory=lambda: dict())


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class TrainerConfig:
    # process_position: int = 0
    num_nodes: int = 1  # number of nodes
    # number of devices, can be auto (lets accelerator finds it), an integer or a list of integer
    devices: int | List[int] | str = 1
    strategy: Optional[str] = None  # ddp, ddp_spawn, or deepspeed ...,etc.
    # TODO custom accelerator to implement
    accelerator: Optional[str] = None  # cpu or gpu or tpu or ...,etc.
    deterministic: bool = False
    max_epochs = 1
    min_epochs = None
    max_steps = - 1
    min_steps = None
    max_time = None
    lr_monitor: Optional[PARAMS] = None
    loggers: Dict[str, PARAMS] | OdnLogger | List[OdnLogger] | bool = False
    model_checkpoint: Optional[PARAMS] = None
    extra_callbacks: Optional[Dict[str, PARAMS]] = None
    extra_params: Optional[PARAMS] = None


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class SeedConfig:
    seed: Optional[int] = None
    seed_worker: bool = True


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class StageConfig:
    stages: STAGES_OR_VALUE | List[STAGES_OR_VALUE] | Dict[STAGES_OR_VALUE, PARAMS] = 'fit'


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class FitConfig:
    """FitApp"""
    model_name: str = DEFAULT_MODEL_NAME
    model_params: PARAMS = field(default_factory=lambda: dict())
    input_name: str = 'input'
    input_params: PARAMS = field(default_factory=lambda: dict())
    stages: STAGES_OR_VALUE | List[STAGES_OR_VALUE] | Dict[STAGES_OR_VALUE, PARAMS] = 'fit'
    # process_position: int = 0
    num_nodes: int = 1  # number of nodes
    # number of devices, can be auto (lets accelerator finds it), an integer or a list of integer
    devices: int | List[int] | str = 1
    strategy: Optional[str] = None  # ddp, ddp_spawn, or deepspeed ...,etc.
    # TODO custom accelerator to implement
    accelerator: Optional[str] = None  # cpu or gpu or tpu or ...,etc.
    deterministic: bool = False
    lr_monitor: Optional[PARAMS] = None
    model_checkpoint: Optional[PARAMS] = None
    loggers: Dict[str, PARAMS] | OdnLogger | List[OdnLogger] | bool = False
    extra_callbacks: Optional[Dict[str, PARAMS]] = None
    seed: Optional[int] = None
    seed_worker: bool = True
    extra_params: Optional[PARAMS] = None
    model_config: Optional[ModelConfig] = None
    input_config: Optional[InputConfig] = None
    trainer_config: Optional[TrainerConfig] = None
    seed_config: Optional[SeedConfig] = None
    stage_config: Optional[StageConfig] = None
    _data: Optional[LightningDataModule] = None
    _model: Optional[LightningModule] = None
    _callbacks: OdnCallback | List[OdnCallback] | None = None
    _trainer: Optional[OdnTrainer] = None
    ###################################
    #        Getter / Setter          #
    ###################################

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: LightningModule):
        self._model = model

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: LightningDataModule):
        self._data = data

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: OdnTrainer):
        self._trainer = trainer

    @property
    def callbacks(self):
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: OdnCallback | List[OdnCallback]):
        self._callbacks = callbacks

    ###################################
    #        Post Init                #
    ###################################
    def __post_init__(self):

        self.seed, self.seed_worker, self.seed_config = FitConfig.seed_everything(seed=self.seed,
                                                                                  seed_worker=self.seed_worker,
                                                                                  seed_config=self.seed_config)
        self._has_fit_stage: bool = False
        self._has_fit_stage, self.stages, self.stage_config = FitConfig.configure_stages(self.stages,
                                                                                         self.stage_config)
        self._model, self.model_config = self.configure_model(model_name=self.model_name,
                                                              model_params=self.model_params,
                                                              model_config=self.model_config)
        self._data, self.input_config = self.configure_input(input_name=self.input_name,
                                                             input_params=self.input_params,
                                                             input_config=self.input_config)
        self.lr_monitor_params = {'logging_interval': 'step'} if self.lr_monitor is None \
            else self.lr_monitor

        self._trainer, self._callbacks, self.trainer_config = self.configure_trainer(
            loggers=self.loggers,
            extra_callbacks=self.extra_callbacks,
            lr_monitor=self.lr_monitor,
            model_checkpoint=self.model_checkpoint,
            num_nodes=self.num_nodes,
            devices=self.devices,
            strategy=self.strategy,
            accelerator=self.accelerator,
            deterministic=self.deterministic,
            extra_params=self.extra_params,
            trainer_config=self.trainer_config
        )
        # _params: Dict[STAGES_OR_VALUE, PARAMS] = field(init=False)
        # _has_fit_stage: bool = field(init=False, default=False)

    ###################################
    #        Methods                  #
    ###################################
    @staticmethod
    def seed_everything(seed: Optional[int] = None,
                        seed_worker: bool = False,
                        seed_config: Optional[SeedConfig] = None) -> Tuple[Optional[int], bool, SeedConfig]:

        if seed_config is not None:
            seed_everything(seed=seed_config.seed, workers=seed_config.seed_worker)
            return seed_config.seed, seed_config.seed_worker, seed_config
        elif seed is not None:
            seed_everything(seed=seed, workers=seed_worker)
            cfg = SeedConfig(seed=seed, seed_worker=seed_worker)
            return seed, seed_worker, cfg
        else:
            return seed, seed_worker, SeedConfig(seed=seed, seed_worker=seed_worker)

    @staticmethod
    def configure_model(model_name: str = DEFAULT_MODEL_NAME,
                        model_params: Optional[PARAMS] = None,
                        model_config: Optional[ModelConfig] = None) -> Tuple[LightningModule, ModelConfig]:
        if model_config is not None:
            instance = MODEL_REGISTRY.create(name=model_config.model_name, **model_config.model_params)
            assert instance is not None
            return instance, model_config
        if model_params is None:
            model_config = ModelConfig(model_name=model_name)
            instance = MODEL_REGISTRY.create(name=model_config.model_name, **model_config.model_params)
            assert instance is not None
            return instance, model_config
        else:
            model_config = ModelConfig(model_name=model_name, model_params=model_params)
            instance = MODEL_REGISTRY.create(name=model_config.model_name, **model_config.model_params)
            assert instance is not None
            return instance, model_config

    @staticmethod
    def configure_input(input_name: str = DEFAULT_INPUT_NAME,
                        input_params: Optional[PARAMS] = None,
                        input_config: Optional[InputConfig] = None) -> Tuple[LightningDataModule, InputConfig]:
        if input_config is not None:
            instance = DATA_REGISTRY.create(name=input_config.input_name,
                                            **input_config.input_params)
            assert instance is not None
            return instance, input_config
        else:
            if input_params:
                instance = DATA_REGISTRY.create(name=input_name,
                                                **input_params)
                assert instance is not None
                return instance, InputConfig(input_name=input_name, input_params=input_params)
            else:
                instance = DATA_REGISTRY.create(name=input_name)
                assert instance is not None
                return instance, InputConfig(input_name=input_name)

    @staticmethod
    def configure_loggers(loggers: Dict[str, PARAMS] | OdnLogger | List[OdnLogger | Dict] | bool
                          ) -> list[OdnLogger] | OdnLogger | bool:
        if loggers is not None:
            return build_loggers(loggers=cast(Dict[str, PARAMS] | OdnLogger | List[OdnLogger | Dict] | bool, loggers))
        else:
            return False

    @staticmethod
    def configure_callbacks(lr_monitor: Optional[PARAMS] = None,
                            model_checkpoint: Optional[PARAMS] = None,
                            extra_callbacks: Optional[Dict[str, PARAMS]] = None
                            ) -> List[OdnCallback]:
        callbacks: List[OdnCallback] = []
        if extra_callbacks is not None:
            callbacks = build_callbacks(callbacks=extra_callbacks)
        if model_checkpoint is not None:
            _model_checkpoint = ModelCheckpoint(**model_checkpoint)
            callbacks.append(_model_checkpoint)
            if lr_monitor:
                _lr_monitor = LearningRateMonitor(**lr_monitor)
                callbacks.append(_lr_monitor)
        return callbacks

    @staticmethod
    def configure_stages(
            stages: STAGES_OR_VALUE | List[STAGES_OR_VALUE] | Dict[STAGES_OR_VALUE, PARAMS],
            stage_config: Optional[StageConfig]
    ) -> Tuple[bool, Union[STAGES_OR_VALUE, List[STAGES_OR_VALUE], Dict[STAGES_OR_VALUE, PARAMS]], StageConfig]:
        _has_fit_stage = False
        if stage_config is not None:
            pass
        else:
            stage_config = StageConfig(stages=stages)
        if isinstance(stages, str):
            if stages in FIT_STAGES:
                _has_fit_stage = True
                stages = cast(List[STAGES_OR_VALUE], [stages])
            else:
                raise MisconfigurationException(message=f"stage {stages} "
                                                        f"should be fit if you don't specify ckpt_path")

        elif isinstance(stages, List):
            if len(set(FIT_STAGES).intersection(set(list(stages)))) <= 0:
                raise MisconfigurationException(message=f"if you use a list of stage, "
                                                        f"you need to declare the {Stages.FIT.value} "
                                                        f"stage. If you don't want to"
                                                        f"declare a {Stages.FIT.value} stage, you need to"
                                                        f"use a dictionary with the ckpt_path parameter"
                                                        f"filled")

            _has_fit_stage = True
            stages = sorted(stages, key=lambda d: STAGE_ORDER[d])
        elif isinstance(stages, Dict):
            for stage in stages.keys():
                if stage in FIT_STAGES:
                    _has_fit_stage = True
            s = sorted(stages, key=lambda d: STAGE_ORDER[d])  # compute sorted keys of Dict in stage order
            stages = {v: stages[v] for v in s}
        # TODO, gives possibility to update parameters by stage
        return _has_fit_stage, stages, stage_config

    @staticmethod
    def configure_trainer(loggers: Dict[str, PARAMS] | OdnLogger | List[OdnLogger] | bool = False,
                          extra_callbacks: Optional[Dict[str, PARAMS]] = None,
                          lr_monitor: Optional[PARAMS] = None,
                          model_checkpoint: Optional[PARAMS] = None,
                          num_nodes: int = 1,
                          devices: int | List[int] | str = 1,
                          strategy: Optional[str] = None,
                          accelerator: Optional[str] = None,
                          deterministic: bool = False,
                          extra_params: Optional[PARAMS] = None,
                          trainer_config: Optional[TrainerConfig] = None
                          ) -> Tuple[OdnTrainer, List[OdnCallback], TrainerConfig]:
        if trainer_config is not None:
            callbacks: Optional[list[OdnCallback]] = FitConfig.configure_callbacks(
                lr_monitor=trainer_config.lr_monitor,
                model_checkpoint=trainer_config.model_checkpoint,
                extra_callbacks=trainer_config.extra_callbacks)

            loggers = trainer_config.loggers
            num_nodes = trainer_config.num_nodes
            devices = trainer_config.devices
            strategy = trainer_config.strategy
            accelerator = trainer_config.accelerator
            deterministic = trainer_config.deterministic
            extra_params = trainer_config.extra_params
        else:
            callbacks = FitConfig.configure_callbacks(lr_monitor=lr_monitor,
                                                      model_checkpoint=model_checkpoint,
                                                      extra_callbacks=extra_callbacks)
            trainer_config = TrainerConfig(num_nodes=num_nodes,
                                           devices=devices,
                                           strategy=strategy,
                                           accelerator=accelerator,
                                           lr_monitor=lr_monitor,
                                           model_checkpoint=model_checkpoint,
                                           extra_callbacks=extra_callbacks,
                                           loggers=loggers,
                                           deterministic=deterministic,
                                           extra_params=extra_params)

        loggers = FitConfig.configure_loggers(loggers=cast(
            Dict[str, PARAMS] | OdnLogger | List[OdnLogger | Dict] | bool, loggers))
        extra_params = dict() if extra_params is None else extra_params
        assert callbacks is not None
        return OdnTrainer(logger=loggers,
                          callbacks=callbacks,
                          accelerator=accelerator,
                          devices=devices,
                          strategy=strategy,
                          deterministic=deterministic,
                          num_nodes=num_nodes,
                          **extra_params
                          ), callbacks, trainer_config


class FitApp(App, metaclass=Singleton):

    def __init__(self, config: FitConfig | Dict):
        super().__init__()
        if isinstance(config, FitConfig):
            self.config = config
        elif isinstance(config, Dict):
            self.config = FitConfig(**config)

    def __call__(self, *args, **kwargs):
        """
        .debug(f'stages: {self.config.stages},\n'
                     f'stages type: {type(self.config.stages)}')
        """
        self.run()

    def run(self):

        if isinstance(self.config.stages, List):
            for stage in self.config.stages:
                self._run_stage(stage=stage)
        else:
            for k, v in self.config.stages.items():
                self._run_stage(stage=k, params=v)

    def _run_stage(self, stage: STAGES_OR_VALUE, params: Optional[PARAMS] = None):
        if stage in FIT_STAGES:
            if params is not None:
                self.config.trainer.fit(model=self.config.model, datamodule=self.config.data, **params)
            else:
                self.config.trainer.fit(model=self.config.model, datamodule=self.config.data)
        elif stage in INFERENCE_STAGES:
            params = params if params is not None else DEFAULT_INFERENCE_PARAMS
            if stage == Stages.VALIDATE or str(Stages.VALIDATE.value):
                self.config.trainer.validate(model=self.config.model, datamodule=self.config.data, **params)
            elif stage == Stages.TEST or str(Stages.TEST.value):
                self.config.trainer.test(model=self.config.model, datamodule=self.config.data, **params)
            else:
                self.config.trainer.predict(model=self.config.model, datamodule=self.config.data, **params)

    @staticmethod
    def get_class_config() -> type:
        return FitConfig
