from torch.utils.data import Dataset
from skimage.util import img_as_float

from odeon.commons.image import image_to_ndarray
from odeon.nn.transforms import ToDoubleTensor


class PatchDataset(Dataset):
    """Dataset based on patch files both images and masks.
    Masks composition must be with of one channel by class.

    Parameters
    ----------
    image_files : list of str
        pathes of image files
    mask_files : list of str
        pathes of image files
    transform : func, optional
        transform function can be one of :class:`Rotation90`, :class:`Radiometry` or :class:`Compose`.
        [albumentation](https://albumentations.readthedocs.io/en/latest/index.html) functions can be used.
        When using :class:`Compose` :class:`ToDoubleTensor` must be added at the end of the transforms list.
        by default None
    width : number, optional
        sample width, if None native width is used, by default None
    height : number, optional
        sample height, if None native height is used, by default None
    image_bands : list of number, optional
        list of band indices to keep in sample generation, by default None
    mask_bands : [type], optional
        list of band indices to keep in sample generation, by default None

    """

    def __init__(self, image_files, mask_files, transform=None, width=None, height=None, image_bands=None,
                 mask_bands=None):

        self.image_files = image_files
        self.image_bands = image_bands
        self.mask_files = mask_files
        self.mask_bands = mask_bands

        self.width = width
        self.height = height
        self.transform_function = transform
        pass

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        # load image file
        image_file = self.image_files[index]
        img = image_to_ndarray(image_file, width=self.width, height=self.height, band_indices=self.image_bands)
        # pixels are normalized to [0, 1]
        img = img_as_float(img)

        # load mask file
        mask_file = self.mask_files[index]
        msk = image_to_ndarray(mask_file, width=self.width, height=self.height)

        sample = {"image": img, "mask": msk}

        # apply transforms
        if self.transform_function is None:
            self.transform_function = ToDoubleTensor()
        sample = self.transform_function(**sample)

        return sample
