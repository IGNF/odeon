import pytorch_lightning.callbacks as C
import pytorch_lightning.loggers as L

from .callbacks import CallbackRegistry
from .logger import LoggerRegistry

LoggerRegistry.register_class(L.MLFlowLogger, name='mlflow')
LoggerRegistry.register_class(L.TensorBoardLogger, name='tensorboard', aliases=['ts_board', 'tsb'])
LoggerRegistry.register_class(L.CometLogger, name='comet')
LoggerRegistry.register_class(L.CSVLogger, name='csv')
LoggerRegistry.register_class(L.WandbLogger, name='wandb')
LoggerRegistry.register_class(L.NeptuneLogger, name='netpune')

CallbackRegistry.register_class(C.Checkpoint, name='checkpoint', aliases=['ckpt'])
CallbackRegistry.register_class(C.LearningRateMonitor, name='lr_monitor', aliases=['lrm'])
CallbackRegistry.register_class(C.ModelCheckpoint, name='model_checkpoint', aliases=['mod_ckpt'])
CallbackRegistry.register_class(C.EarlyStopping, name='early_stopping', aliases=['early_stop', 'e_stop'])
CallbackRegistry.register_class(C.RichModelSummary, name='rich_model_summary', aliases=['rich_sum'])
CallbackRegistry.register_class(C.QuantizationAwareTraining, name='quantization')
CallbackRegistry.register_class(C.StochasticWeightAveraging, name='stochastic_weighted_averaging', aliases=['swa'])
CallbackRegistry.register_class(C.ModelPruning, name='model_pruning', aliases=['pruning'])
CallbackRegistry.register_class(C.GradientAccumulationScheduler, name='gradient_accumulator_scheduler',
                                aliases=['grad_acc_sched', 'gas'])
CallbackRegistry.register_class(C.TQDMProgressBar, name='tqdm_progress_bar', aliases=['tqdm'])
