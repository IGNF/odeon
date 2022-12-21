from logging import getLogger

import torch

from odeon.core.metric import MetricRegistry
from odeon.core.registry import GenericRegistry

logger = getLogger(__name__)


def test_generic_registry():

    dummy_class_name = 'dummy_class'
    aliases = ['du', 'dum']

    @GenericRegistry.register(name='dummy_class', aliases=aliases)
    class MyDummyClass:
        def __init__(self, param1: int, param2: str):
            self.param1 = param1,
            self.param2 = param2
    kwargs = {'param1': 3, 'param2': 'foo'}
    my_dummy_1: MyDummyClass = GenericRegistry.create(dummy_class_name, **kwargs)
    my_dummy_2: MyDummyClass = MyDummyClass(**kwargs)
    assert my_dummy_1.param1 == my_dummy_2.param1
    assert my_dummy_1.param2 == my_dummy_2.param2


def test_generic_metric_registry():

    from torchmetrics.classification import BinaryAccuracy
    bin_acc: BinaryAccuracy = BinaryAccuracy()
    cl_name = 'binary_accuracy'
    aliases = ['b_acc', 'binary_acc']
    pred = torch.tensor([1, 0, 0, 1])
    target = torch.tensor([1, 1, 1, 1])
    MetricRegistry.register_class(cl=BinaryAccuracy, name=cl_name, aliases=aliases)
    bin_acc_fact = MetricRegistry.create(name=cl_name, aliases=aliases)
    assert isinstance(bin_acc_fact, BinaryAccuracy)
    assert bin_acc(pred, target) == bin_acc_fact(pred, target)
