import pytorch_lightning.callbacks as C
import pytorch_lightning.loggers as L

from odeon.core.app import APP_REGISTRY
from odeon.core.plugins.plugin import OdnPlugin, PluginMaturity
from odeon.core.registry import GENERIC_REGISTRY

from .app import FitApp
from .callbacks import CALLBACK_REGISTRY
from .logger import LOGGER_REGISTRY
from .trainer import OdnTrainer

__all__ = ['pl_logger_plugin', 'pl_callback_plugin',
           'fit_plugin', 'FitApp',
           'OdnTrainer', 'APP_REGISTRY',
           'CALLBACK_REGISTRY', 'LOGGER_REGISTRY']

"""APP PLUGIN
"""
fit_plugin = OdnPlugin(name='fit',
                       author='samy KHELIFI-RICHARDS',
                       plugin_maturity=str(PluginMaturity.DEVELOPMENT.value),
                       version='0.1',
                       elements={'fit': {'registry': APP_REGISTRY, 'class': FitApp}})

"""LOGGER PLUGIN
"""
pl_logger_plugin = OdnPlugin(name='pl_logger',
                             author='samy KHELIFI-RICHARDS',
                             plugin_maturity=str(PluginMaturity.DEVELOPMENT.value),
                             version='0.1',
                             elements={'logger_registry': {'class': LOGGER_REGISTRY, 'registry': GENERIC_REGISTRY,
                                                           'aliases': ['logger_r']},
                                       'mlflow': {'registry': LOGGER_REGISTRY, 'class': L.MLFlowLogger},
                                       'tensorboard': {'registry': LOGGER_REGISTRY, 'class': L.TensorBoardLogger},
                                       'comet': {'registry': LOGGER_REGISTRY, 'class': L.CometLogger},
                                       'csv': {'registry': LOGGER_REGISTRY, 'class': L.CSVLogger},
                                       'wandb': {'registry': LOGGER_REGISTRY, 'class': L.WandbLogger},
                                       'neptune': {'registry': LOGGER_REGISTRY, 'class': L.NeptuneLogger}})

"""CALLBACK PLUGIN
"""
pl_callback_plugin = OdnPlugin(name='pl_callback',
                               author='samy KHELIFI-RICHARDS',
                               plugin_maturity=str(PluginMaturity.DEVELOPMENT.value),
                               version='0.1',
                               elements={'callback_registry': {'class': CALLBACK_REGISTRY,
                                                               'registry': GENERIC_REGISTRY,
                                                               'aliases': ['callback_r']},
                                         'checkpoint': {'registry': CALLBACK_REGISTRY, 'class': C.Checkpoint,
                                                        'aliases': ['ckpt']},
                                         'lr_monitor': {'registry': CALLBACK_REGISTRY, 'class': C.LearningRateMonitor,
                                                        'aliases': ['lrm']},
                                         'model_checkpoint': {'registry': CALLBACK_REGISTRY,
                                                              'class': C.ModelCheckpoint,
                                                              'aliases': ['mod_ckpt']},
                                         'early_stopping': {'registry': CALLBACK_REGISTRY,
                                                            'class': C.EarlyStopping,
                                                            'aliases': ['mod_ckpt']},
                                         'rich_model_summary': {'registry': CALLBACK_REGISTRY,
                                                                'class': C.RichModelSummary,
                                                                'aliases': ['mod_ckpt']},
                                         'stochastic_weighted_averaging': {'registry': CALLBACK_REGISTRY,
                                                                           'class': C.StochasticWeightAveraging,
                                                                           'aliases': ['swa']},
                                         'model_pruning': {'registry': CALLBACK_REGISTRY, 'class': C.ModelPruning,
                                                           'aliases': ['pruning']},
                                         'gradient_accumulator_scheduler': {'registry': CALLBACK_REGISTRY,
                                                                            'class': C.GradientAccumulationScheduler,
                                                                            'aliases': ['grad_acc_sched', 'gas']},
                                         'tqdm_progress_bar': {'registry': CALLBACK_REGISTRY,
                                                               'class': C.TQDMProgressBar}})
