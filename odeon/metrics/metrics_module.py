import numpy as np
import torch
from torchmetrics import Metric
from odeon.commons.metric.metrics_multiclass import torch_metrics_from_cm
from torchmetrics.functional import confusion_matrix


class OdeonMetrics(Metric):

    def __init__(self,
                 num_classes,
                 class_labels,
                 dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.class_labels = class_labels
        default = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.add_state("confmat", default=default, dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        self.confmat += confusion_matrix(preds,
                                         target,
                                         num_classes=self.num_classes,
                                         normalize=None)

    def compute(self):
        metrics_collecton = torch_metrics_from_cm(cm_macro=self.confmat,
                                                  class_labels=self.class_labels)
        return metrics_collecton


class MeanVector(Metric):
    def __init__(self,
                 len_vector,
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.len_vector = len_vector
        default = torch.zeros(len_vector, dtype=torch.float32)
        self.add_state("sum_acc", default=default, dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor):
        self.sum_acc += value
        self.weight += 1

    def compute(self):
        return self.sum_acc / self.weight


class WellfordVariance(Metric):
    """
        Compute variance in one pass with Wellford's algorithm.
    """

    def __init__(self,
                 len_vector,
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.len_vector = len_vector
        default = torch.zeros(len_vector, dtype=torch.float32)
        self.add_state("means", default=default, dist_reduce_fx="sum")
        self.add_state("m2", default=default, dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor):
        self.weight += 1
        delta = torch.mean(value - self.means, axis=0)
        self.means += delta / self.weight
        delta2 = torch.mean(value - self.means, axis=0)
        self.m2 += delta * delta2

    def compute(self):
        if self.weight < 2:
            return float("nan")
        else:
            variance = self.m2 / (self.weight - 1)
            return variance


class IncrementalVariance(Metric):
    """
        Compute the variance in one pass with an incremental averaging of the mean and variance computed for each sample.
    """
    def __init__(self,
                 len_vector,
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.len_vector = len_vector
        default = torch.zeros(len_vector, dtype=torch.float32)
        self.add_state("avg_value", default=default, dist_reduce_fx="sum")
        self.add_state("avg_value_squared", default=default, dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @staticmethod
    def lerp_function(avg_value, new_value,  alpha):
        return avg_value * (1 - alpha) + new_value * alpha

    def update(self, value: torch.Tensor):
        self.avg_value = self.lerp_function(self.avg_value, value, 1 / (self.weight + 1))
        self.avg_value_squared = self.lerp_function(self.avg_value_squared, value * value, 1 / (self.weight + 1))
        self.weight += 1

    def compute(self):
        return torch.mean(torch.abs(self.avg_value_squared - (self.avg_value * self.avg_value)), axis=0)
