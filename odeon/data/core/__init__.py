import albumentations as A

from .transform import TRANSFORM_REGISTRY

TRANSFORM_REGISTRY.register_class(A.VerticalFlip, name='vertical_flip', aliases=['v_flip'])
TRANSFORM_REGISTRY.register_class(A.HorizontalFlip, name='horizontal_flip', aliases=['h_flip'])
TRANSFORM_REGISTRY.register_class(A.Transpose, name='transpose')
TRANSFORM_REGISTRY.register_class(A.Rotate, name='rotate', aliases=['rot'])
TRANSFORM_REGISTRY.register_class(A.RandomRotate90, name='random_rotate_90', aliases=['rrotate90', 'rrot90'])
TRANSFORM_REGISTRY.register_class(A.resize, name='resize')
TRANSFORM_REGISTRY.register_class(A.RandomCrop, name='random_crop', aliases=['r_crop'])
TRANSFORM_REGISTRY.register_class(A.RandomResizedCrop, name='random_resize_crop', aliases=['rr_crop'])
TRANSFORM_REGISTRY.register_class(A.RandomSizedCrop, name='random_sized_crop', aliases=['rs_crop'])
TRANSFORM_REGISTRY.register_class(A.gaussian_blur, name='gaussian_blur', aliases=['g_blur'])
TRANSFORM_REGISTRY.register_class(A.gauss_noise, name='gauss_noise', aliases=['g_noise'])
TRANSFORM_REGISTRY.register_class(A.ColorJitter, name='color_jitter', aliases=['c_jitter', 'c_jit'])
TRANSFORM_REGISTRY.register_class(A.RandomGamma, name='random_gamma', aliases=['r_gamma'])
