from hydra.core.config_store import ConfigStore
from odeon.configs.core import (
    Files,
    DataModuleConf,
    TransformsConf,
    TrainConfig
)
from odeon.configs.structured.optimizers import (
    AdadeltaConf,
    AdagradConf,
    AdamaxConf,
    AdamConf,
    AdamWConf,
    ASGDConf,
    LBFGSConf,
    RMSpropConf,
    RpropConf,
    SGDConf,
    SparseAdamConf
)
from odeon.configs.structured.schedulers import (
    CosineAnnealingLRConf,
    CosineAnnealingWarmRestartsConf,
    CyclicLRConf,
    ExponentialLRConf,
    MultiStepLRConf,
    OneCycleLRConf,
    ReduceLROnPlateauConf,
    StepLRConf
)
from odeon.configs.structured.profilers import (
    AdvancedProfilerConf, 
    PassThroughProfilerConf,
    SimpleProfilerConf
)
from odeon.configs.structured.losses import (
    BCEWithLogitsLossConf,
    CrossEntropyWithLogitsLossConf,
    ComboLossConf,
    SoftDiceLossConf,
    DiceLossConf,
    JaccardLossConf,
    LovaszLossConf,
    FocalLoss2dConf
)
from odeon.configs.structured.callbacks import (
    EarlyStoppingConf,
    GPUStatsMonitorConf,
    GradientAccumulationSchedulerConf,
    LearningRateMonitorConf,
    TimerConf,
    ProgressBarConf,
    TQDMProgressBarConf,
    ModelCheckpointConf,
    LogConfusionMatrixConf,
    MetricsWandbConf,
    UploadCodeAsArtifactConf,
    LightningCheckpointConf,
    HistorySaverConf,
    CustomPredictionWriterConf,
    ContinueTrainingConf,
    ExoticCheckPointConf,
    MetricsAdderConf,
    HParamsAdderConf,
    GraphAdderConf,
    HistogramAdderConf,
    PredictionsAdderConf
)
from odeon.configs.structured.loggers import (
    CSVLoggerConf,
    TensorBoardLoggerConf,
    WandbLoggerConf
)
# from odeon.configs.structured.metrics import (
#     AccuracyConf,
#     AUCConf,
#     AUROCConf,
#     AveragePrecisionConf,
#     ConfusionMatrixConf,
#     F1Conf,
#     FBetaConf,
#     HammingDistanceConf,
#     IoUConf,
#     PrecisionConf,
#     PrecisionRecallCurveConf,
#     RecallConf,
#     ROCConf,
#     StatScoresConf,
#     OdeonMetricsConf,
#     MeanVectorConf,
#     IncrementalVarianceConf,
#     WellfordVarianceConf
# )
from odeon.configs.structured.model import (
    UnetConf,
    LightUnetOdeonConf,
    DeepLabOdeonConf,
    UnetSmpConf,
    DeepLabV3SmpConf,
    DeepLabV3PlusSmpConf,
    FPNSmpConf,
    LinknetSmpConf,
    MAnetSmpConf,
    PANSmpConf,
    PSPNetSmpConf
)
from odeon.configs.trainer import LightningTrainerConf

# Get config store instance
config_store = ConfigStore.instance()

# Stores main config into the repository
config_store.store(name='train_config', node=TrainConfig)

# Store core classes structured schema
config_store.store(group='files', name='default', node=Files)
config_store.store(group='datamodule', name='datamodule', node=DataModuleConf)
config_store.store(group='transforms', name='transforms', node=TransformsConf)
config_store.store(group='trainer', name='default', node=LightningTrainerConf)

# Store optimizers
config_store.store(group='optimizer', name='adadelta', node=AdadeltaConf)
config_store.store(group='optimizer', name='adagrad', node=AdagradConf)
config_store.store(group='optimizer', name='adamax', node=AdamaxConf)
config_store.store(group='optimizer', name='adam', node=AdamConf)
config_store.store(group='optimizer', name='adamw', node=AdamWConf)
config_store.store(group='optimizer', name='asgd', node=ASGDConf)
config_store.store(group='optimizer', name='lbfgs', node=LBFGSConf)
config_store.store(group='optimizer', name='rms_prop', node=RMSpropConf)
config_store.store(group='optimizer', name='r_prop', node=RpropConf)
config_store.store(group='optimizer', name='sgd', node=SGDConf)
config_store.store(group='optimizer', name='sparse_adam', node=SparseAdamConf)

# Store schedulers 
config_store.store(group='scheduler', name='cosine_annealing_lr', node=CosineAnnealingLRConf)
config_store.store(group='scheduler', name='cosine_annealing_warm_restarts', node=CosineAnnealingWarmRestartsConf)
config_store.store(group='scheduler', name='cyclic_lr', node=CyclicLRConf)
config_store.store(group='scheduler', name='exponential_lr', node=ExponentialLRConf)
config_store.store(group='scheduler', name='multi_step_lr', node=MultiStepLRConf)
config_store.store(group='scheduler', name='one_cycle_lr', node=OneCycleLRConf)
config_store.store(group='scheduler', name='reduce_lr_on_plateau', node=ReduceLROnPlateauConf)
config_store.store(group='scheduler', name='step_lr', node=StepLRConf)

# Store profilers
config_store.store(group='profiler', name='advanced', node=AdvancedProfilerConf)
config_store.store(group='profiler', name='pass_through', node=PassThroughProfilerConf)
config_store.store(group='profiler', name='simple', node=SimpleProfilerConf)

# Store losses
config_store.store(group='loss', name='bce_with_logits', node=BCEWithLogitsLossConf)
config_store.store(group='loss', name='ce_with_logits', node=CrossEntropyWithLogitsLossConf)
config_store.store(group='loss', name='combo', node=ComboLossConf)
config_store.store(group='loss', name='soft_dice', node=SoftDiceLossConf)
config_store.store(group='loss', name='dice', node=DiceLossConf)
config_store.store(group='loss', name='jaccard', node=JaccardLossConf)
config_store.store(group='loss', name='lovasz', node=LovaszLossConf)
config_store.store(group='loss', name='focal', node=FocalLoss2dConf)

# Store loggers
config_store.store(group='logger', name='csv', node=CSVLoggerConf)
config_store.store(group='logger', name='tensorboard', node=TensorBoardLoggerConf)
config_store.store(group='logger', name='wandb', node=WandbLoggerConf)

# Store callbacks
config_store.store(group='callbacks', name='early_stop', node=EarlyStoppingConf)
config_store.store(group='callbacks', name='gpu_stats_monitor', node=GPUStatsMonitorConf)
config_store.store(group='callbacks', name='grad_acc_scheduler', node=GradientAccumulationSchedulerConf)
config_store.store(group='callbacks', name='lr_monitor', node=LearningRateMonitorConf)
config_store.store(group='callbacks', name='model_checkpoint', node=ModelCheckpointConf)
config_store.store(group='callbacks', name='timer', node=TimerConf)
config_store.store(group='callbacks', name='progress_bar', node=ProgressBarConf)
config_store.store(group='callbacks', name='tqdm_progress_bar', node=TQDMProgressBarConf)
config_store.store(group='callbacks', name='wandb_cm', node=LogConfusionMatrixConf)
config_store.store(group='callbacks', name='wandb_metrics', node=MetricsWandbConf)
config_store.store(group='callbacks', name='wandb_log_code', node=UploadCodeAsArtifactConf)
config_store.store(group='callbacks', name='custom_ckpt', node=LightningCheckpointConf)
config_store.store(group='callbacks', name='history_saver', node=HistorySaverConf)
config_store.store(group='callbacks', name='pred_writer', node=CustomPredictionWriterConf)
config_store.store(group='callbacks', name='continue_training', node=ContinueTrainingConf)
config_store.store(group='callbacks', name='exotic_ckpt', node=ExoticCheckPointConf)
config_store.store(group='callbacks', name='tensorboard_metrics', node=MetricsAdderConf)
config_store.store(group='callbacks', name='tensorboard_hparams', node=HParamsAdderConf)
config_store.store(group='callbacks', name='tensorboard_graph', node=GraphAdderConf)
config_store.store(group='callbacks', name='tensorboard_histogram', node=HistogramAdderConf)
config_store.store(group='callbacks', name='tensorboard_predictions', node=PredictionsAdderConf)

# Store metrics
# config_store.store(group='metrics', name='accuracy', node=AccuracyConf)
# config_store.store(group='metrics', name='auc', node=AUCConf)
# config_store.store(group='metrics', name='auroc', node=AUROCConf)
# config_store.store(group='metrics', name='avg_precision', node=AveragePrecisionConf)
# config_store.store(group='metrics', name='confusion_matrix', node=ConfusionMatrixConf)
# config_store.store(group='metrics', name='f1_score', node=F1Conf)
# config_store.store(group='metrics', name='f_beta', node=FBetaConf)
# config_store.store(group='metrics', name='hamming', node=HammingDistanceConf)
# config_store.store(group='metrics', name='iou', node=IoUConf)
# config_store.store(group='metrics', name='precision', node=PrecisionConf)
# config_store.store(group='metrics', name='pr_curve', node=PrecisionRecallCurveConf)
# config_store.store(group='metrics', name='reall', node=RecallConf)
# config_store.store(group='metrics', name='history_saver', node=HistorySaverConf)
# config_store.store(group='metrics', name='roc', node=ROCConf)
# config_store.store(group='metrics', name='history_saver', node=HistorySaverConf)
# config_store.store(group='metrics', name='stats_score', node=StatScoresConf)
# config_store.store(group='metrics', name='odeon_metrics', node=OdeonMetricsConf)
# config_store.store(group='metrics', name='mean_vector', node=MeanVectorConf)
# config_store.store(group='metrics', name='incremental_variance', node=IncrementalVarianceConf)
# config_store.store(group='metrics', name='wellford_variance', node=WellfordVarianceConf)

# Store models
config_store.store(group='model', name='unet', node=UnetConf)
config_store.store(group='model', name='lightunet', node=LightUnetOdeonConf)
config_store.store(group='model', name='deeplab', node=DeepLabOdeonConf)
config_store.store(group='model', name='unet_smp', node=UnetSmpConf)
config_store.store(group='model', name='deeplabv3_smp', node=DeepLabV3SmpConf)
config_store.store(group='model', name='deeplavv3plus_smp', node=DeepLabV3PlusSmpConf)
config_store.store(group='model', name='fpn_smp', node=FPNSmpConf)
config_store.store(group='model', name='linknet_smp', node=LinknetSmpConf)
config_store.store(group='model', name='manet_smp', node=MAnetSmpConf)
config_store.store(group='model', name='pan_smp', node=PANSmpConf)
config_store.store(group='model', name='pspnet_smp', node=PSPNetSmpConf)

# Store transforms
from odeon.configs.structured.transforms import (
    BlurConf,
    CLAHEConf,
    ChannelDropoutConf,
    ChannelShuffleConf,
    ColorJitterConf,
    DownscaleConf,
    EmbossConf,
    EqualizeConf,
    FDAConf,
    FancyPCAConf,
    FromFloatConf,
    GaussNoiseConf,
    GaussianBlurConf,
    GlassBlurConf,
    HistogramMatchingConf,
    HueSaturationValueConf,
    ISONoiseConf,
    InvertImgConf,
    MedianBlurConf,
    MotionBlurConf,
    MultiplicativeNoiseConf,
    NormalizeConf,
    PosterizeConf,
    RGBShiftConf,
    RandomBrightnessContrastConf,
    RandomFogConf,
    RandomGammaConf,
    RandomRainConf,
    RandomShadowConf,
    RandomSnowConf,
    RandomSunFlareConf,
    SharpenConf,
    SolarizeConf,
    SuperpixelsConf,
    ToFloatConf,
    ToGrayConf,
    ToSepiaConf,
    AffineConf,
    CenterCropConf,
    CoarseDropoutConf,
    CropAndPadConf,
    CropConf,
    CropNonEmptyMaskIfExistsConf,
    ElasticTransformConf,
    FlipConf,
    GridDistortionConf,
    GridDropoutConf,
    HorizontalFlipConf,
    LambdaConf,
    LongestMaxSizeConf,
    MaskDropoutConf,
    NoOpConf,
    OpticalDistortionConf,
    PadIfNeededConf,
    PerspectiveConf,
    PiecewiseAffineConf,
    RandomCropConf,
    RandomCropNearBBoxConf,
    RandomGridShuffleConf,
    RandomResizedCropConf,
    RandomRotate90Conf,
    RandomScaleConf,
    RandomSizedBBoxSafeCropConf,
    RandomSizedCropConf,
    ResizeConf,
    RotateConf,
    SafeRotateConf,
    ShiftScaleRotateConf,
    SmallestMaxSizeConf,
    TransposeConf,
    VerticalFlipConf,
)