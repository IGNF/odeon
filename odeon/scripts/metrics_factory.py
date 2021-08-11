from bc_metrics import BC_Metrics
from mc_1l_metrics import MC_1L_Metrics
from mc_ml_metrics import MC_ML_Metrics


def Metrics_Factory(type_classifier):

    metrics = {"Binary case": BC_Metrics,
               "Multi-class mono-label": MC_1L_Metrics,
               "Multi-class multi-label": MC_ML_Metrics}

    return metrics[type_classifier]
