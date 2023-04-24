from typing import cast

import pytorch_lightning.callbacks as C
import pytorch_lightning.loggers as L

from .callbacks import CALLBACK_REGISTRY
from .logger import LOGGER_REGISTRY

LOGGER_REGISTRY.register_class(cast(L.Logger, L.MLFlowLogger), name='mlflow')
LOGGER_REGISTRY.register_class(cast(L.Logger, L.TensorBoardLogger), name='tensorboard', aliases=['ts_board', 'tsb'])
LOGGER_REGISTRY.register_class(cast(L.Logger, L.CometLogger), name='comet')
LOGGER_REGISTRY.register_class(cast(L.Logger, L.CSVLogger), name='csv')
LOGGER_REGISTRY.register_class(cast(L.Logger, L.WandbLogger), name='wandb')
LOGGER_REGISTRY.register_class(cast(L.Logger, L.NeptuneLogger), name='netpune')

CALLBACK_REGISTRY.register_class(cast(C.Callback, C.Checkpoint), name='checkpoint', aliases=['ckpt'])
CALLBACK_REGISTRY.register_class(cast(C.Callback, C.LearningRateMonitor), name='lr_monitor', aliases=['lrm'])
CALLBACK_REGISTRY.register_class(cast(C.Callback, C.ModelCheckpoint), name='model_checkpoint', aliases=['mod_ckpt'])
CALLBACK_REGISTRY.register_class(cast(C.Callback, C.EarlyStopping), name='early_stopping', aliases=['early_stop',
                                                                                                    'e_stop'])
CALLBACK_REGISTRY.register_class(cast(C.Callback, C.RichModelSummary), name='rich_model_summary', aliases=['rich_sum'])
CALLBACK_REGISTRY.register_class(cast(C.Callback, C.QuantizationAwareTraining), name='quantization')
CALLBACK_REGISTRY.register_class(cast(C.Callback, C.StochasticWeightAveraging), name='stochastic_weighted_averaging',
                                 aliases=['swa'])
CALLBACK_REGISTRY.register_class(cast(C.Callback, C.ModelPruning), name='model_pruning', aliases=['pruning'])
CALLBACK_REGISTRY.register_class(cast(C.Callback, C.GradientAccumulationScheduler),
                                 name='gradient_accumulator_scheduler',
                                 aliases=['grad_acc_sched', 'gas'])
CALLBACK_REGISTRY.register_class(cast(C.Callback, C.TQDMProgressBar), name='tqdm_progress_bar', aliases=['tqdm'])
