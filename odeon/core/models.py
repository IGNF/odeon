from abc import ABC
from typing import Callable, List, Optional, Protocol, Tuple, Union

import torch.nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor

from .types import OdnMetric


class MetricInterface(Protocol):

    def update_metrics(self, preds: Tensor, targets: Tensor, metric: OdnMetric):
        ...

    def compute_metrics(self, preds: Tensor, targets: Tensor, metric: OdnMetric):
        ...

    def configure_metrics(self, *args, **kwargs) -> Tuple[Optional[OdnMetric], Optional[OdnMetric],
                                                          Optional[OdnMetric]]:
        ...


class MetricMixin(ABC, MetricInterface):
    """Class implementing data and behaviour of Metric computing in OdnModule
    """

    def __init__(self,
                 clone_fit_on_val: bool = False,
                 clone_fit_on_test: bool = False,
                 clone_val_on_test: bool = False,
                 *args,
                 **kwargs):

        self._clone_fit_on_val = clone_fit_on_val
        self._clone_fit_on_test = clone_fit_on_test
        self._clone_val_on_test = clone_val_on_test
        self._fit_metrics: Optional[OdnMetric]
        self._val_metrics: Optional[OdnMetric]
        self._test_metrics: Optional[OdnMetric]
        self._fit_metrics, self._val_metrics, self._test_metrics = self.configure_metrics()

    @property
    def fit_metrics(self):
        return self._fit_metrics

    @property
    def val_metrics(self):
        return self._val_metrics

    @property
    def test_metrics(self):
        return self._test_metrics


class TranferLearningInterface(Protocol):

    # TODO implement Interface
    @staticmethod
    def transfer_from(source: torch.nn.Module,
                      target: torch.nn.Module,
                      module_list: List) -> torch.nn.Module:
        """
        algo:
            for key, value in module_dict.items(
        Parameters
        ----------
        target

        Returns
        -------

        """
        ...


class TransferLearningMixin(ABC, TranferLearningInterface):
    # TODO implement MixinClass
    def __init__(self):
        ...

    def transfer_from(source: torch.nn.Module, target: torch.nn.Module, module_list: List[str]) -> torch.nn.Module:

        for value in module_list:
            ...
        return torch.nn.Module[torch.nn.Identity]


class OdnModel(LightningModule, MetricMixin):

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super(LightningModule).__init__()
        super(MetricMixin).__init__(**self._model_params)
        self._model: torch.nn.Module
        self._loss: Callable

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def configure_models(self, *args, **kwargs):
        ...

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        ...

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        ...

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        ...

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        ...
