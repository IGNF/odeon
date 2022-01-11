from odeon.commons.metric.metrics_binary import MetricsBinary
from odeon.commons.metric.metrics_multiclass import MetricsMulticlass


def MetricsFactory(type_classifier):

    metrics = {"binary": MetricsBinary,
               "multiclass": MetricsMulticlass}

    return metrics[type_classifier]
