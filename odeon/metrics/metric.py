from typing import Dict, List

from odeon.core.logger import get_logger
from odeon.core.python_env import debug_mode
from odeon.core.registry import GenericRegistry

from .types import OdnMetric

logger = get_logger(__name__, debug=debug_mode)


class BinaryMetricRegistry(GenericRegistry[type[OdnMetric]]):
    _registry: Dict[str, type[OdnMetric]] = {}
    _alias_registry: Dict[str, str] = {}

    @classmethod
    def create(cls, name: str, **kwargs) -> OdnMetric:
        """
        Factory command to create an instance.
        This method gets the appropriate Registered class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.

        Parameters
        ----------
         name: str, The name of the executor to create.
         kwargs

        Returns
        -------
         Callable: An instance of the executor that is created.
        """

        if name not in cls._registry:
            logger.error(f"{name} not registered in registry {str(name)}")
            raise KeyError()

        _class = cls.get(name=name)
        return _class(**kwargs)


class MulticlassMetricRegistry(GenericRegistry[type[OdnMetric]]):
    _registry: Dict[str, type[OdnMetric]] = {}
    _alias_registry: Dict[str, str] = {}

    @classmethod
    def create(cls, name: str, **kwargs) -> OdnMetric:
        """
        Factory command to create an instance.
        This method gets the appropriate Registered class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.

        Parameters
        ----------
         name: str, The name of the executor to create.
         kwargs

        Returns
        -------
         Callable: An instance of the executor that is created.
        """

        if name not in cls._registry:
            logger.error(f"{name} not registered in registry {str(name)}")
            raise KeyError()

        _class = cls.get(name=name)
        return _class(**kwargs)


class MultilabelMetricRegistry(GenericRegistry[type[OdnMetric]]):
    _registry: Dict[str, type[OdnMetric]] = {}
    _alias_registry: Dict[str, str] = {}

    @classmethod
    def create(cls, name: str, **kwargs) -> OdnMetric:
        """
        Factory command to create an instance.
        This method gets the appropriate Registered class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.

        Parameters
        ----------
         name: str, The name of the executor to create.
         kwargs

        Returns
        -------
         Callable: An instance of the executor that is created.
        """

        if name not in cls._registry:
            logger.error(f"{name} not registered in registry {str(name)}")
            raise KeyError()

        _class = cls.get(name=name)
        return _class(**kwargs)


BINARY_METRIC_REGISTRY = BinaryMetricRegistry
MULTICLASS_METRIC_REGISTRY = MulticlassMetricRegistry
MULTILABEL_METRIC_REGISTRY = MultilabelMetricRegistry


def build_binary_metrics(metrics: List[Dict]) -> List[OdnMetric]:
    result: List[OdnMetric] = list()
    for metric in metrics:
        name = metric['name']
        if 'params' in metric:
            params: Dict = metric['params']
            result.append(BINARY_METRIC_REGISTRY.create(name=name, **params))
        else:
            result.append(BINARY_METRIC_REGISTRY.create(name=name))
    return result


def build_multiclass_metrics(metrics: List[Dict]) -> List[OdnMetric]:
    result: List[OdnMetric] = list()
    for metric in metrics:
        name = metric['name']
        if 'params' in metric:
            params: Dict = metric['params']
            result.append(MULTICLASS_METRIC_REGISTRY.create(name=name, **params))
        else:
            result.append(MULTICLASS_METRIC_REGISTRY.create(name=name))
    return result


def build_multilabel_metrics(metrics: List[Dict]) -> List[OdnMetric]:
    result: List[OdnMetric] = list()
    for metric in metrics:
        name = metric['name']
        if 'params' in metric:
            params: Dict = metric['params']
            result.append(MULTILABEL_METRIC_REGISTRY.create(name=name, **params))
        else:
            result.append(MULTILABEL_METRIC_REGISTRY.create(name=name))
    return result
