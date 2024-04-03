from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from pytorch_lightning.utilities import CombinedLoader
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
    """
        A container for storing all necessary components for a single stage of data processing in
        machine learning workflows. It encapsulates a DataLoader for batched data processing, a Dataset
        for raw data management, an optional transformation function for data augmentation or preprocessing,
        and a DataFrame for additional metadata or annotations related to the dataset.

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader object for iterating over the dataset in batches.
        dataframe : DATAFRAME
            A DataFrame containing metadata or annotations for the dataset. The specific type of
            DATAFRAME should be defined according to the project's requirements, typically Pandas DataFrame.
        dataset : Dataset
            The Dataset object containing the raw data to be processed. This could be any object that
            inherits from PyTorch's Dataset class.
        transform : Optional[Callable], optional
            An optional callable (function or object) for performing transformations or augmentations on
            the data. This could be a composition of several transformations implemented with libraries
            such as torchvision or albumentations. Defaults to None.

        Attributes
        ----------
        dataloader : DataLoader
            Provides an iterable over the dataset for batched processing.
        dataframe : DATAFRAME
            Stores metadata or annotations associated with the dataset.
        dataset : Dataset
            Holds the raw data for processing.
        transform : Optional[Callable]
            A function or callable object for data transformation.

        Examples
        --------
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> import pandas as pd
        >>> import torch

        # Create a simple dataset and dataloader for demonstration
        >>> dataset = TensorDataset(torch.randn(100, 2), torch.randint(0, 2, (100,)))
        >>> dataloader = DataLoader(dataset, batch_size=10)

        # Example dataframe for metadata or annotations
        >>> dataframe = pd.DataFrame({'id': range(100), 'label': torch.randint(0, 2, (100,)).numpy()})

        # Instantiate the Data class without transformations
        >>> data_instance = Data(dataloader=dataloader, dataframe=dataframe, dataset=dataset, transform=None)

        # Accessing the DataLoader and DataFrame
        >>> for batch in data_instance.dataloader:
        ...     # Process each batch
        ...     pass
        >>> print(data_instance.dataframe.head())

        Notes
        -----
        This class is designed to be immutable, meaning that once an instance is created, its attributes
        cannot be modified. This design choice helps to ensure the consistency of data throughout the
        processing stages.
        """
    dataloader: DataLoader
    dataframe: DATAFRAME
    dataset: Dataset
    transform: Optional[Callable]


class Input(OdnData):
    """
        A PyTorch Lightning Module specialized for handling georeferenced data across various stages
        of machine learning workflows, including fitting, validation, testing, and prediction, in the most genereic
        way (multi instance, and coming soon multimodal with remote sensing time series and lidar. It is
        capable of processing data with optional transformations and organizing data loaders for each
        respective stage.

        The `Input` class inherits from `odeon.data.core.OdnData`, integrating closely with the PyTorch Lightning
        framework to facilitate efficient data handling and transformations within a geospatial
        context.

        Parameters
        ----------
        fit_params : List[Dict] | Dict | None, optional
            Parameters for setting up the data during the fitting stage. It could be a list of
            dictionaries, each representing parameters for a separate data instance, or a single
            dictionary if only one instance is used. Defaults to None.
        validate_params : List[Dict] | Dict | None, optional
            Parameters for setting up the data during the validation stage, structured similarly
            to `fit_params`. Defaults to None.
        test_params : List[Dict] | Dict | None, optional
            Parameters for setting up the data during the testing stage. Defaults to None.
        predict_params : List[Dict] | Dict | None, optional
            Parameters for setting up the data during the prediction stage. Defaults to None.


        Methods
        -------
        setup(stage: Optional[str] = None) -> None
            Prepares data for the specified stage. If `stage` is None, it sets up data for all stages
            based on the provided parameters.
        train_dataloader() -> TRAIN_DATALOADERS
            Returns a dataloader or a combined dataloader for the training data.
        val_dataloader() -> EVAL_DATALOADERS
            Returns a dataloader or a combined dataloader for the validation data.
        test_dataloader() -> EVAL_DATALOADERS
            Returns a dataloader or a combined dataloader for the test data.
        predict_dataloader() -> EVAL_DATALOADERS
            Returns a dataloader or a combined dataloader for the prediction data.

        Notes
        -----
        - This class requires specific parameters for setting up data for different stages, which are
          encapsulated in `Data` instances. Each `Data` instance includes a `DataLoader`, a dataset,
          an optional transformation function, and a dataframe.
        - The actual data preparation logic, including instantiation of `Data` objects, is handled by
          the `DataFactory.build_data` method, which is not detailed in this documentation.

        Examples
        --------
        >>> fit_params = [{'data_param1': value1, 'data_param2': value2}, {'data_param1': value3}]
        >>> validate_params = {'data_param1': value5, 'data_param2': value6}
        >>> input_module = Input(fit_params=fit_params, validate_params=validate_params)
        >>> input_module.setup('fit')
        >>> train_loader = input_module.train_dataloader()
        """

    def __init__(self,
                 fit_params: List[Dict] | Dict | None = None,
                 validate_params: List[Dict] | Dict | None = None,
                 test_params: List[Dict] | Dict | None = None,
                 predict_params: List[Dict] | Dict | None = None):

        super(Input, self).__init__()
        self.fit_params = fit_params
        self.validate_params = validate_params
        self.test_params = test_params
        self.predict_params = predict_params
        self._fit: Data | Dict[str, Data] | None = None
        self._validate: Data | Dict[str, Data] | None = None
        self._test: Data | Dict[str, Data] | None = None
        self._predict: Data | Dict[str, Data] | None = None

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
            assert self.test_params, (f'you want to run a stage {stage} but you have not filled {stage}_params :'
                                      f'{self.test_params}')
            self._test = {f'test-{i + 1}': Input._instantiate_data(params=params, stage=Stages.TEST) for i, params
                          in enumerate(list(self.test_params))} if isinstance(self.test_params, List)\
                else Input._instantiate_data(params=dict(self.test_params), stage=Stages.TEST)
        if stage == Stages.PREDICT.value or stage == Stages.PREDICT:
            assert self.predict_params, f'you want to run a stage {stage} but you have not filled {stage}_params :' \
                                        f'{self.predict_params}'
            self._predict = {f'predict-{i + 1}': Input._instantiate_data(params=params, stage=Stages.PREDICT)
                             for i, params in enumerate(list(self.predict_params))} \
                if isinstance(self.predict_params, List) else Input._instantiate_data(params=dict(self.predict_params),
                                                                                      stage=Stages.PREDICT)

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
    def test(self) -> Data | Dict[str, Data] | None:
        return self._test

    @property
    def predict(self) -> Data | Dict[str, Data] | None:
        return self._predict

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if isinstance(self._fit, Data | dict):
            if isinstance(self._fit, Dict):
                d: Dict[str, DataLoader] = {k: v.dataloader for k, v in self._fit.items()}
                return CombinedLoader(d, mode='min_size')
            else:
                return self._fit.dataloader
        else:
            raise NotImplementedError('fit is not implemented yet, fit is None')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if isinstance(self._validate, Data | dict):
            if isinstance(self._validate, Dict):
                d: Dict[str, DataLoader] = {k: v.dataloader for k, v in self._validate.items()}
                return CombinedLoader(d, mode='min_size')
            else:
                return self._validate.dataloader
        else:
            raise NotImplementedError('validate is not implemented yet, validate is None')

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if isinstance(self._test, Data | dict):
            if isinstance(self._test, Dict):
                d: Dict[str, DataLoader] = {k: v.dataloader for k, v in self._test.items()}
                return CombinedLoader(d, mode='min_size')
            else:
                return self._test.dataloader
        else:
            raise NotImplementedError('test is not implemented yet, test is None')

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if isinstance(self._predict, Data | dict):
            if isinstance(self._predict, Dict):
                d: Dict[str, DataLoader] = {k: v.dataloader for k, v in self._predict.items()}
                return CombinedLoader(d, mode='min_size')
            else:
                return self._predict.dataloader
        else:
            raise NotImplementedError('predict is not implemented yet, predict is None')
