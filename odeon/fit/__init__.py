# from typing import cast

import pytorch_lightning.callbacks as C
import pytorch_lightning.loggers as L

from odeon.core.app import APP_REGISTRY
from odeon.core.plugins.plugin import OdnPlugin

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
fit_plugin = OdnPlugin(elements={'fit': {'registry': APP_REGISTRY, 'class': FitApp}})

"""LOGGER PLUGIN
"""
pl_logger_plugin = OdnPlugin(elements={'mlflow': {'registry': LOGGER_REGISTRY, 'class': L.MLFlowLogger},
                                       'tensorboard': {'registry': LOGGER_REGISTRY, 'class': L.TensorBoardLogger},
                                       'comet': {'registry': LOGGER_REGISTRY, 'class': L.CometLogger},
                                       'csv': {'registry': LOGGER_REGISTRY, 'class': L.CSVLogger},
                                       'wandb': {'registry': LOGGER_REGISTRY, 'class': L.WandbLogger},
                                       'neptune': {'registry': LOGGER_REGISTRY, 'class': L.NeptuneLogger}})

"""CALLBACK PLUGIN
"""
pl_callback_plugin = OdnPlugin(elements={'checkpoint': {'registry': CALLBACK_REGISTRY, 'class': C.Checkpoint,
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
                                         'quantization': {'registry': CALLBACK_REGISTRY,
                                                          'class': C.QuantizationAwareTraining},
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