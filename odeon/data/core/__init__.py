import albumentations as A

from .transform import TransformRegistry

TransformRegistry.register_class(A.VerticalFlip, name='vertical_flip', aliases=['v_flip'])
TransformRegistry.register_class(A.HorizontalFlip, name='horizontal_flip', aliases=['h_flip'])
TransformRegistry.register_class(A.Transpose, name='transpose')
TransformRegistry.register_class(A.Rotate, name='rotate', aliases=['rot'])
TransformRegistry.register_class(A.RandomRotate90, name='random_rotate_90', aliases=['rrotate90', 'rrot90'])
TransformRegistry.register_class(A.resize, name='resize')
TransformRegistry.register_class(A.RandomCrop, name='random_crop', aliases=['r_crop'])
TransformRegistry.register_class(A.RandomResizedCrop, name='random_resize_crop', aliases=['rr_crop'])
TransformRegistry.register_class(A.RandomSizedCrop, name='random_sized_crop', aliases=['rs_crop'])
TransformRegistry.register_class(A.gaussian_blur, name='gaussian_blur', aliases=['g_blur'])
TransformRegistry.register_class(A.gauss_noise, name='gauss_noise', aliases=['g_noise'])
TransformRegistry.register_class(A.ColorJitter, name='color_jitter', aliases=['c_jitter', 'c_jit'])
TransformRegistry.register_class(A.RandomGamma, name='random_gamma', aliases=['r_gamma'])
