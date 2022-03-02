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
