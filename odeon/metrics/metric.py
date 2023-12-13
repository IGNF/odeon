from typing import Dict, List

from odeon.core.registry import GenericRegistry

from .types import OdnMetric


class BinaryMetricRegistry(GenericRegistry[OdnMetric]):
    _registry: Dict[str, OdnMetric] = {}
    _alias_registry: Dict[str, str] = {}

class MulticlassMetricRegistry(GenericRegistry[OdnMetric]):
    _registry: Dict[str, OdnMetric] = {}
    _alias_registry: Dict[str, str] = {}

class MultilabelMetricRegistry(GenericRegistry[OdnMetric]):
    _registry: Dict[str, OdnMetric] = {}
    _alias_registry: Dict[str, str] = {}

BINARY_METRIC_REGISTRY = BinaryMetricRegistry
MULTICLASS_METRIC_REGISTRY = MulticlassMetricRegistry
MULTILABEL_METRIC_REGISTRY = MultilabelMetricRegistry


def build_metrics(metrics: List[Dict]) -> List[OdnMetric]:
    result: List[OdnMetric] = list()
    for metric in metrics:
        name = metric['name']
        if 'params' in metric:
            params: Dict = metric['params']
            BINARY_METRIC_REGISTRY.create(name=name, **params)
        else:
            BINARY_METRIC_REGISTRY.create(name=name)
    return result
