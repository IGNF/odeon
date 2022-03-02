import numpy as np
import torch
from torchmetrics import Metric
from odeon.commons.metric.metrics_multiclass import torch_get_metrics_from_cm
from torchmetrics.functional import confusion_matrix as torch_cm


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
        self.confmat += torch_cm(preds,
                                 target,
                                 num_classes=self.num_classes,
                                 normalize=None)

    def compute(self):

        cm_macro = self.confmat
        metrics_by_class, metrics_micro, _, cm_micro = \
            torch_get_metrics_from_cm(cm_macro=cm_macro,
                                      nbr_class=self.num_classes,
                                      class_labels=self.class_labels,
                                      weighted=False,
                                      weights=None)

        metrics = {'cm_macro': cm_macro,
                   'cm_micro': cm_micro,
                   'Overall/Accuracy': metrics_micro['Precision'],
                   'Overall/IoU': metrics_micro['IoU']}

        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU', 'Specificity']:
            counter= []
            for class_i in self.class_labels:
                metrics[class_i + '/' + metric] = metrics_by_class[class_i][metric]
                counter.append(metrics_by_class[class_i][metric])
            metrics['Average/' + metric] = torch.Tensor(counter).mean()

        return metrics