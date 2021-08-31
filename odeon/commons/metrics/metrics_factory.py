from odeon.commons.metrics.metrics_binary import Metrics_Binary
from odeon.commons.metrics.metrics_multiclass import Metrics_Multiclass


def Metrics_Factory(type_classifier):

    metrics = {"binary": Metrics_Binary,
               "multiclass": Metrics_Multiclass}

    return metrics[type_classifier.lower()]
