from typing import Dict, List

from odeon.core.registry import GenericRegistry
from odeon.core.types import OdnMetric


class MetricRegistry(GenericRegistry[OdnMetric]):
    _registry: Dict[str, OdnMetric] = {}


METRIC_REGISTRY = MetricRegistry
GenericRegistry.register_class(cl=METRIC_REGISTRY, name='metrics', aliases=['metric_registry'])


def build_metrics(metrics: List[Dict]) -> List[OdnMetric]:
    result: List[OdnMetric] = list()
    for metric in metrics:
        name = metric['name']
        if 'params' in metric:
            params: Dict = metric['params']
            METRIC_REGISTRY.create(name=name, **params)
        else:
            METRIC_REGISTRY.create(name=name)
    return result
