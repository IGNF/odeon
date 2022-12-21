from typing import Dict, List, Type

from torchmetrics import Metric, MetricCollection

"""
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC,
                                         BinaryCalibrationError,
                                         BinaryConfusionMatrix, BinaryF1Score,
                                         BinaryJaccardIndex, BinaryPrecision,
                                         BinaryRecall, BinaryROC,
                                         BinarySpecificity, MulticlassAccuracy,
                                         MulticlassAUROC,
                                         MulticlassCalibrationError,
                                         MulticlassConfusionMatrix,
                                         MulticlassF1Score,
                                         MulticlassJaccardIndex,
                                         MulticlassPrecision, MulticlassRecall,
                                         MulticlassROC, MultilabelAccuracy,
                                         MultilabelAUROC, MultilabelF1Score,
                                         MultilabelPrecision, MultilabelRecall,
                                         MultilabelROC)
"""
from odeon.core.registry import GenericRegistry
from odeon.core.types import OdnMetric


@GenericRegistry.register('MetricRegistry', aliases=['metricReg'])
class MetricRegistry(GenericRegistry[Type[OdnMetric]]):
    @classmethod
    def register_fn(cls, cl: Type[OdnMetric], name: str):
        assert issubclass(cl, Metric) or issubclass(cl, MetricCollection)
        cls._registry[name] = cl


def build_metrics(metrics: List[Dict]):
    for metric in metrics:
        name = metric['name']
        if 'params' in metric:
            params: Dict = metric['params']
            MetricRegistry.create(name=name, **params)
        else:
            MetricRegistry.create(name=name)
