from abc import ABC
from typing import Dict, Optional, Protocol, Tuple

from torch import Tensor

from .types import OdnMetric

METRIC_TUPLE = Tuple[Optional[Dict[str, OdnMetric]], Optional[Dict[str, OdnMetric]], Optional[Dict[str, OdnMetric]]]


class MetricInterface(Protocol):

    def update(self, preds: Tensor, targets: Tensor, metric: OdnMetric):
        ...

    def compute(self, preds: Tensor, targets: Tensor, metric: OdnMetric):
        ...

    def configure(self, *args, **kwargs) -> METRIC_TUPLE:
        ...

    def log(self):
        ...


class MetricMixin(ABC):
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
        self._fit_metrics: Optional[OdnMetric | Dict[str, OdnMetric]] = None
        self._val_metrics: Optional[OdnMetric | Dict[str, OdnMetric]] = None
        self._test_metrics: Optional[OdnMetric | Dict[str, OdnMetric]] = None
        self._fit_metrics, self._val_metrics, self._test_metrics = self.configure()

    @property
    def fit_metrics(self):
        return self._fit_metrics

    @property
    def val_metrics(self):
        return self._val_metrics

    @property
    def test_metrics(self):
        return self._test_metrics

    def configure(self) -> METRIC_TUPLE:
        ...
