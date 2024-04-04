import albumentations as A

from odeon.core.plugins.plugin import OdnPlugin, PluginMaturity
from odeon.core.registry import GENERIC_REGISTRY

from .core.registry import DATA_REGISTRY
from .core.transform import (ALBU_TRANSFORM_REGISTRY, ONE_OFF_ALIASES,
                             ONE_OFF_NAME)
from .data_module import Input

__all__ = ['data_plugin', 'albu_transform_plugin', 'Input', 'DATA_REGISTRY', 'ALBU_TRANSFORM_REGISTRY']


"""
ALBU TRANSFORM REGISTRY
"""
albu_transform_plugin = OdnPlugin(name='albu_transform',
                                  author='samy KHELIFI-RICHARDS',
                                  plugin_maturity=str(PluginMaturity.DEVELOPMENT.value),
                                  version='0.1',
                                  elements={'albu_registry': {'class': ALBU_TRANSFORM_REGISTRY,
                                                              'registry': GENERIC_REGISTRY,
                                                              'aliases': ['albu_r']},
                                            'vertical_flip': {'class': A.VerticalFlip,
                                                              'registry': ALBU_TRANSFORM_REGISTRY,
                                                              'aliases': ['v_flip']},
                                            'horizontal_flip': {'class': A.HorizontalFlip,
                                                                'registry': ALBU_TRANSFORM_REGISTRY,
                                                                'aliases': ['h_flip']},
                                            'transpose': {'class': A.Transpose, 'registry': ALBU_TRANSFORM_REGISTRY},
                                            'rotate': {'class': A.Rotate,
                                                       'registry': ALBU_TRANSFORM_REGISTRY, 'aliases': ['rot']},
                                            'random_rotate_90': {'class': A.RandomRotate90,
                                                                 'registry': ALBU_TRANSFORM_REGISTRY,
                                                                 'aliases': ['v_flip']},
                                            'resize': {'class': A.resize,
                                                       'registry': ALBU_TRANSFORM_REGISTRY},
                                            'random_crop': {'class': A.RandomCrop,
                                                            'aliases': ['r_crop'], 'registry': ALBU_TRANSFORM_REGISTRY},
                                            'random_resize_crop': {'class': A.RandomResizedCrop,
                                                                   'aliases': ['rr_crop'],
                                                                   'registry': ALBU_TRANSFORM_REGISTRY},
                                            'random_sized_crop': {'class': A.RandomSizedCrop,
                                                                  'aliases': ['rs_crop'],
                                                                  'registry': ALBU_TRANSFORM_REGISTRY},
                                            'gaussian_blur': {'class': A.gaussian_blur,
                                                              'aliases': ['g_blur'],
                                                              'registry': ALBU_TRANSFORM_REGISTRY},
                                            'gaussian_noise': {'class': A.gauss_noise,
                                                               'aliases': ['g_noise', 'gaussian_noise'],
                                                               'registry': ALBU_TRANSFORM_REGISTRY},
                                            'color_jitter': {'class': A.RandomCrop,
                                                             'aliases': ['r_crop'],
                                                             'registry': ALBU_TRANSFORM_REGISTRY},
                                            'random_gamma': {'class': A.ColorJitter,
                                                             'aliases': ['c_jitter', 'c_jit'],
                                                             'registry': ALBU_TRANSFORM_REGISTRY},
                                            ONE_OFF_NAME: {'class': A.OneOf,
                                                           'aliases': ONE_OFF_ALIASES,
                                                           'registry': ALBU_TRANSFORM_REGISTRY}
                                            })


"""DATA PLUGIN
"""
data_plugin = OdnPlugin(name='input',
                        author='samy KHELIFI-RICHARDS',
                        plugin_maturity=str(PluginMaturity.DEVELOPMENT.value),
                        version='0.1',
                        elements={'data_registry': {'class': DATA_REGISTRY,
                                                    'registry': GENERIC_REGISTRY,
                                                    'aliases': ['data_reg']},
                                  'input': {'class': Input,
                                            'registry': DATA_REGISTRY,
                                            'aliases': ['default_input', 'universal_data_module']}}
                        )
