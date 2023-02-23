from dataclasses import dataclass, field
from typing import Dict, List, Optional, cast

from pytorch_lightning import LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from odeon.core.app import App
from odeon.core.types import PARAMS, STAGES_OR_VALUE, OdnLogger, OdnCallback, Stages
from odeon.models.core.models import ModelRegistry
from odeon.data.core.data import DataRegistry
from odeon.core.exceptions import MisconfigurationException

from .callbacks import build_callbacks
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


@dataclass
class FitConfig:
    """FitApp"""
    model_name: str
    model_params: PARAMS = field(default_factory=PARAMS)
    _model: LightningModule = field(init=False)
    input_name: str = 'input'
    input_params: PARAMS = field(default_factory=PARAMS)
    _data: LightningDataModule = field(init=False)
    _trainer: OdnTrainer = field(init=False)
    stages: STAGES_OR_VALUE | List[STAGES_OR_VALUE] | Dict[STAGES_OR_VALUE] = 'fit'
    _has_fit_stage: bool = field(init=False, default=False)
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
    loggers: Optional[Dict[str, PARAMS] | OdnLogger | List[OdnLogger]] = None
    _callbacks: OdnCallback | List[OdnCallback] = field(init=False)
    extra_callbacks: Optional[Dict[str, PARAMS]] = None
    seed: Optional[int] = None
    seed_worker: bool = True
    extra_params: Optional[PARAMS] = None
    _params: Dict[STAGES_OR_VALUE, PARAMS] = field(init=False)

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
    def params(self):
        return self._params

    @params.setter
    def params(self, params: Dict[STAGES_OR_VALUE, PARAMS] ):
        self._params = params

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
        self.seed_everything()
        self.configure_stages()
        self.configure_loggers()
        self.configure_model()
        self.configure_input()
        self.lr_monitor_params = {'logging_interval': 'step'} if self.lr_monitor is None \
            else self.lr_monitor
        self.callbacks = self.configure_callbacks()
        self.configure_trainer()

    ###################################
    #        Functions                #
    ###################################

    def seed_everything(self):
        if self.seed is not None:
            seed_everything(seed=self.seed, workers=self.seed_worker)

    def configure_model(self):
        self.model = ModelRegistry.create(name=self.model_name, **self.model_params)

    def configure_input(self):
        self.data = DataRegistry.create(name=self.input_name, **self.input_params)

    def configure_loggers(self):
        if self.loggers is not None:
            self.loggers = build_loggers(loggers=self.loggers)

    def configure_callbacks(self) -> List[OdnCallback]:
        callbacks: List[OdnCallback] = []
        if self.extra_callbacks is not None:
            callbacks = build_callbacks(callbacks=self.extra_callbacks)
        if self.model_checkpoint is not None:
            model_checkpoint = ModelCheckpoint(**self.model_checkpoint)
            callbacks.append(model_checkpoint)
        lr_monitor = LearningRateMonitor(**self.lr_monitor_params)
        callbacks.append(lr_monitor)
        return callbacks

    def configure_stages(self):
        if isinstance(self.stages, str):
            if self.stages in FIT_STAGES:
                self._has_fit_stage = True
                self.stages = [cast(STAGES_OR_VALUE, self.stages)]
            else:
                raise MisconfigurationException(message=f"stage {self.stages} "
                                                        f"should be fit if you don't specify ckpt_path")
        elif isinstance(self.stages, List):
            if len(set(FIT_STAGES).intersection(set(list(self.stages)))) <= 0:
                raise MisconfigurationException(message=f"if you use a list of stage, "
                                                        f"you need to declare the {Stages.FIT.value} "
                                                        f"stage. If you don't want to"
                                                        f"declare a {Stages.FIT.value} stage, you need to"
                                                        f"use a dictionary with the ckpt_path parameter"
                                                        f"filled")
            self._has_fit_stage = True
            self.stages = sorted(self.stages, key=lambda d: STAGE_ORDER[d])
        elif isinstance(self.stages, Dict):
            for stage in self.stages.keys():
                if stage in FIT_STAGES:
                    self._has_fit_stage = True
            s = sorted(self.stages, key=lambda d: STAGE_ORDER[d])  # compute sorted keys of Dict in stage order
            self.stages: Dict[STAGES_OR_VALUE, PARAMS] = {v: self.stages[v] for v in s}
        # TODO, gives possibility to update parameters by stage

    def configure_trainer(self):
        self.trainer = OdnTrainer(logger=self.loggers,
                                  callbacks=self.callbacks,
                                  accelerator=self.accelerator,
                                  devices=self.devices,
                                  strategy=self.strategy,
                                  deterministic=self.deterministic,
                                  num_nodes=self.num_nodes,
                                  **self.extra_params
                                  )  # TODO ends instantiation


class FitApp(App):

    def __init__(self, config: FitConfig):

        super().__init__()
        self.config = config

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
