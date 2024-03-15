from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data import DataLoader, Dataset

from odeon.core.app_utils import Stages
from odeon.core.logger import get_logger
from odeon.core.types import DATAFRAME, STAGES_OR_VALUE
from odeon.data.stage import DataFactory

from .core.types import OdnData

logger = get_logger(__name__)
STAGES_D = {stage: stage.value for stage in Stages}
REVERSED_STAGES_D = {stage.value: stage for stage in Stages}


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class Data:
    dataloader: DataLoader
    dataframe: DATAFRAME
    dataset: Dataset
    transform: Optional[Callable]


class Input(OdnData):
    """Input DataModule
    Take a

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self,
                 fit_params: List[Dict] | Dict | None = None,
                 validate_params: List[Dict] | Dict | None = None,
                 test_params: Dict = None,
                 predict_params: Dict = None):

        super(Input, self).__init__()
        self.fit_params = fit_params
        self.validate_params = validate_params
        self.test_params = test_params
        self.predict_params = predict_params
        self._fit: Data | Dict[str, Data] | None = None
        self._validate: Data | Dict[str, Data] | None = None
        self._test: Data | None = None
        self._predict: Data | None = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == Stages.FIT.value or stage == Stages.FIT:
            assert self.fit_params, f'you want to run a stage {stage} but you have not filled {stage}_params :' \
                                    f'{self.fit_params}'
            self._fit = {f'fit-{i+1}': Input._instantiate_data(params=params, stage=Stages.FIT)
                         for i, params in enumerate(list(self.fit_params))} if isinstance(self.fit_params, List)\
                else Input._instantiate_data(params=dict(self.fit_params), stage=Stages.FIT)
            if self.validate_params:
                self._validate = {f'validate-{i + 1}': Input._instantiate_data(params=params, stage=Stages.VALIDATE)
                                  for i, params in enumerate(list(self.validate_params))} \
                    if isinstance(self.validate_params, List) else Input._instantiate_data(
                    params=dict(self.validate_params),
                    stage=Stages.VALIDATE)
            else:
                logger.warning('you are setting up a fit stage without having configured the validation')

        if stage == Stages.VALIDATE.value or stage == Stages.VALIDATE:
            assert self.validate_params, f'you want to run a stage {stage} but you have not filled {stage}_params :' \
                                         f'{self.validate_params}'
            self._validate = {f'validate-{i + 1}': Input._instantiate_data(params=params, stage=Stages.VALIDATE)
                              for i, params in enumerate(list(self.validate_params))} \
                if isinstance(self.validate_params, List)\
                else Input._instantiate_data(params=dict(self.validate_params), stage=Stages.VALIDATE)
        if stage == Stages.TEST.value or stage == Stages.TEST:
            assert self.test_params, f'you want to run a stage {stage} but you have not filled {stage}_params :' \
                                     f'{self.test_params}'
            self._test = Input._instantiate_data(params=self.test_params, stage=Stages.TEST)
        if stage == Stages.PREDICT.value or stage == Stages.PREDICT:
            assert self.predict_params, f'you want to run a stage {stage} but you have not filled {stage}_params :' \
                                        f'{self.predict_params}'
            self._predict = Input._instantiate_data(params=self.predict_params, stage=Stages.PREDICT)

    @staticmethod
    def _instantiate_data(params: Dict, stage: STAGES_OR_VALUE) -> Data:
        logger.info(f'params: {params}')
        params['stage'] = stage
        dataloader, dataset, transform, dataframe = DataFactory.build_data(**params)
        data = Data(dataloader=dataloader,
                    dataset=dataset,
                    transform=transform,
                    dataframe=dataframe)
        return data

    @property
    def fit(self) -> Data | Dict[str, Data] | None:
        return self._fit

    @property
    def validate(self) -> Data | Dict[str, Data] | None:
        return self._validate

    @property
    def test(self) -> Data | None:
        return self._test

    @property
    def predict(self) -> Data | None:
        return self._predict

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if isinstance(self._fit, Data):
            if isinstance(self._fit, Dict):
                d: Dict[str, DataLoader] = {k: v.dataloader for k, v in self._fit.items()}
                return d
            else:
                return self._fit.dataloader
        else:
            raise NotImplementedError('fit is not implemented yet, fit is None')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if isinstance(self._validate, Data):
            if isinstance(self._validate, Dict):
                l: List = [v.dataloader for k, v in self._validate.items()]
                return l
            else:
                return self._validate.dataloader
        else:
            raise NotImplementedError('validate is not implemented yet, validate is None')

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if isinstance(self._test, Data):
            return self._test.dataloader
        else:
            raise NotImplementedError('test is not implemented yet, test is None')

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if isinstance(self._predict, Data):
            return self._predict.dataloader
        else:
            raise NotImplementedError('predict is not implemented yet, predict is None')
