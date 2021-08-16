from metrics_binary import Metrics_Binary
from metrics_multiclass import Metrics_Multiclass


def Metrics_Factory(type_classifier):

    metrics = {"Binary": Metrics_Binary,
               "Multiclass": Metrics_Multiclass}

    return metrics[type_classifier]
