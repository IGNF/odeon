import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data import DataLoader, Dataset

from odeon.core.runner_utils import Stages
from odeon.core.types import DATAFRAME, STAGES_OR_VALUE

from .stage import DataFactory

logger = logging.getLogger(__name__)
STAGES_D = {stage: stage.value for stage in Stages}
REVERSED_STAGES_D = {stage.value: stage for stage in Stages}


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class Data:
    dataloader: DataLoader
    dataframe: DATAFRAME
    dataset: Dataset
    transform: Optional[Callable]


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class Input(LightningDataModule):
    """Input DataModule
    Take a

    Attributes
    ----------

    Methods
    -------

    """

    fit_params: Union[List[Dict], Dict] = None
    validate_params: Dict = None
    test_params: Dict = None
    predict_params: Dict = None

    _fit: Union[Data, Dict[str, Data]] = field(init=False)
    _validate: Data = field(init=False)
    _test: Data = field(init=False)
    _predict: Data = field(init=False)

    def __post_init__(self):

        if self.fit_params:
            self._fit = {f'fit-{i+1}': self._instanciate_data(params=params, stage=Stages.FIT)
                         for i, params in enumerate(list(self.fit_params))} if isinstance(self.fit_params, List)\
                else self._instanciate_data(params=dict(self.fit_params), stage=Stages.FIT)
        self._validate = self._instanciate_data(params=self.validate_params, stage=Stages.VALIDATE) \
            if self.validate_params else None
        self._test = self._instanciate_data(params=self.test_params, stage=Stages.TEST) \
            if self.test_params else None
        self._predict = self._instanciate_data(params=self.predict_params, stage=Stages.PREDICT) \
            if self.predict_params else None

    def _instanciate_data(self, params: Dict, stage: STAGES_OR_VALUE) -> Data:
        logger.info(f'params: {params}')
        params['stage'] = stage
        dataloader, dataset, transform, dataframe = DataFactory.build_data(**params)
        data = Data(dataloader=dataloader,
                    dataset=dataset,
                    transform=transform,
                    dataframe=dataframe)
        return data

    @property
    def fit(self) -> Union[Data, Dict[str, Data]]:
        return self._fit

    @property
    def validate(self) -> Data:
        return self._validate

    @property
    def test(self) -> Data:
        return self._test

    @property
    def predict(self) -> Data:
        return self._predict

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        if isinstance(self._fit, Dict):
            d: Dict[str, DataLoader] = {k: v.dataloader for k, v in self._fit.items()}
            return d
        else:
            return self._fit.dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._validate.dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._test.dataloader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._predict.dataloader
