from pydantic.dataclasses import dataclass
from typing import Optional, List
from omegaconf import MISSING


# Custom Odeon metrics
@dataclass
class OdeonMetricsConf:
    _target__ = "odeon.modules.metrics_module.OdeonMetrics"
    num_classes : int = MISSING 
    class_labels: List[str] = MISSING
    dist_sync_on_step : Optional[bool] = False


@dataclass
class MeanVectorConf:
    _target__ = "odeon.modules.metrics_module.MeanVector"
    len_vector : int = MISSING 
    dist_sync_on_step : Optional[bool] = False


@dataclass
class IncrementalVarianceConf:
    _target__ = "odeon.modules.metrics_module.IncrementalVariance"
    len_vector : int = MISSING 
    dist_sync_on_step : Optional[bool] = False


@dataclass
class WellfordVarianceConf:
    _target__ = "odeon.modules.metrics_module.WellfordVariance"
    len_vector : int = MISSING 
    dist_sync_on_step : Optional[bool] = False


# Torchmetrics metrics/classes
@dataclass
class AccuracyConf:
    _target_: str = 'torchmetrics.classification.Accuracy'
    threshold: float = 0.5
    num_classes: Optional[int] = None
    average: str = 'micro'
    mdmc_average: Optional[str] = 'global'
    ignore_index: Optional[int] = None
    top_k: Optional[int] = None
    multiclass: Optional[bool] = None
    subset_accuracy: bool = False
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None,
    # dist_sync_fn: Callable = None,


@dataclass
class AUCConf:
    _target_: str = 'torchmetrics.classification.AUC'
    reorder: bool = False
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO


@dataclass
class AUROCConf:
    _target_: str = 'torchmetrics.classification.AUROC'
    num_classes: Optional[int] = None
    pos_label: Optional[int] = None
    average: Optional[str] = 'macro'
    max_fpr: Optional[float] = None
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO


@dataclass
class AveragePrecisionConf:
    _target_: str = 'torchmetrics.classification.AveragePrecision'
    num_classes: Optional[int] = None
    pos_label: Optional[int] = None
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO


@dataclass
class ConfusionMatrixConf:
    _target_: str = 'torchmetrics.classification.ConfusionMatrix'
    num_classes: int = MISSING
    normalize: Optional[str] = None
    threshold: float = 0.5
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO


@dataclass
class F1Conf:
    _target_: str = 'torchmetrics.classification.F1'
    num_classes: Optional[int] = None
    threshold: float = 0.5
    average: str = 'micro'
    mdmc_average: Optional[str] = None
    ignore_index: Optional[int] = None
    top_k: Optional[int] = None
    multiclass: Optional[bool] = None
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None,
    # dist_sync_fn: Callable = None,


@dataclass
class FBetaConf:
    _target_: str = 'torchmetrics.classification.FBeta'
    num_classes: Optional[int] = None
    beta: float = 1.0
    threshold: float = 0.5
    average: str = 'micro'
    mdmc_average: Optional[str] = None
    ignore_index: Optional[int] = None
    top_k: Optional[int] = None
    multiclass: Optional[bool] = None
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None,
    # dist_sync_fn: Callable = None,


@dataclass
class HammingDistanceConf:
    _target_: str = 'torchmetrics.classification.HammingDistance'
    threshold: float = 0.5
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO


@dataclass
class IoUConf:
    _target_: str = 'torchmetrics.classification.IoU'
    num_classes: int = MISSING
    ignore_index: Optional[int] = None
    absent_score: float = 0.0
    threshold: float = 0.5
    reduction: str = 'elementwise_mean'
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO


@dataclass
class PrecisionConf:
    _target_: str = 'torchmetrics.classification.Precision'
    num_classes: Optional[int] = None
    threshold: float = 0.5
    average: str = 'micro'
    mdmc_average: Optional[str] = None
    ignore_index: Optional[int] = None
    top_k: Optional[int] = None
    multiclass: Optional[bool] = None
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO


@dataclass
class PrecisionRecallCurveConf:
    _target_: str = 'torchmetrics.classification.PrecisionRecallCurve'
    num_classes: Optional[int] = None
    pos_label: Optional[int] = None
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO


@dataclass
class RecallConf:
    _target_: str = 'torchmetrics.classification.Recall'
    num_classes: Optional[int] = None
    threshold: float = 0.5
    average: str = 'micro'
    mdmc_average: Optional[str] = None
    ignore_index: Optional[int] = None
    top_k: Optional[int] = None
    multiclass: Optional[bool] = None
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO


@dataclass
class ROCConf:
    _target_: str = 'torchmetrics.classification.ROC'
    num_classes: Optional[int] = None
    pos_label: Optional[int] = None
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO


@dataclass
class StatScoresConf:
    _target_: str = 'torchmetrics.classification.StatScores'
    threshold: float = 0.5
    top_k: Optional[int] = None
    reduce: str = 'micro'
    num_classes: Optional[int] = None
    ignore_index: Optional[int] = None
    mdmc_reduce: Optional[str] = None
    multiclass: Optional[bool] = None
    compute_on_step: bool = True
    dist_sync_on_step: bool = False
    # process_group: Optional[Any] = None  # TODO
    # dist_sync_fn: Callable = None  # TODO
