from odeon.core.plugins.plugin import OdnPlugin

from .change.module.change_unet import ChangeUnet
from .core.models import MODEL_REGISTRY
from .segmentation.segmentation import SegmentationModule

__all__ = ['ChangeUnet', 'SegmentationModule', 'model_plugin', 'MODEL_REGISTRY']

model_plugin = OdnPlugin(elements={'change_unet': {'registry': MODEL_REGISTRY, 'class': ChangeUnet,
                                                   'aliases': ['cunet']},
                                   'segmentation': {'registry': MODEL_REGISTRY, 'class': SegmentationModule,
                                                    'aliases': ['psm']}}
                         )
