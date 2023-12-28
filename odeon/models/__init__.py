from odeon.core.plugins.plugin import OdnPlugin, PluginMaturity
from odeon.core.registry import GENERIC_REGISTRY

from .change.module.change_unet import ChangeUnet
from .core.models import MODEL_REGISTRY
from .segmentation.segmentation import SegmentationModule

__all__ = ['ChangeUnet', 'SegmentationModule', 'model_plugin', 'MODEL_REGISTRY']

model_plugin = OdnPlugin(name='models',
                         author='samy KHELIFI-RICHARDS',
                         plugin_maturity=str(PluginMaturity.DEVELOPMENT.value),
                         version='0.1',
                         elements={'model_registry': {'class': MODEL_REGISTRY, 'registry': GENERIC_REGISTRY,
                                                      'aliases': ['model_r', 'models_r']},
                                   'change_unet': {'registry': MODEL_REGISTRY, 'class': ChangeUnet,
                                                   'aliases': ['cunet']},
                                   'segmentation': {'registry': MODEL_REGISTRY, 'class': SegmentationModule,
                                                    'aliases': ['psm']}}
                         )
