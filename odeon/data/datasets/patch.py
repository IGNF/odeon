import os
from torch.utils.data import Dataset
from skimage.util import img_as_float
import rasterio
# from rasterio.plot import reshape_as_raster
import numpy as np
from odeon import LOGGER
from odeon.commons.image import raster_to_ndarray, CollectionDatasetReader
from odeon.data.transforms.base import ToDoubleTensor, ToPatchTensor, ToWindowTensor
from odeon.commons.rasterio import affine_to_ndarray
from odeon.commons.folder_manager import create_folder
from odeon.commons.exception import OdeonError, ErrorCodes


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

    def __init__(
    self,
    image_files, 
    mask_files, 
    transform=None, 
    width=None, 
    height=None, 
    image_bands=None,             
    mask_bands=None,
    get_sample_info=False
    ):
        self.image_files = image_files
        self.image_bands = image_bands
        self.mask_files = mask_files
        self.mask_bands = mask_bands
        self.width = width
        self.height = height
        self.transform_function = transform
        self.get_sample_info = get_sample_info

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        # load image file
        image_file = self.image_files[index]
        img, meta = raster_to_ndarray(
                                    image_file,
                                    width=self.width,
                                    height=self.height,
                                    resolution=None,
                                    band_indices=self.image_bands
                                    )
        
        sample = {"image": img}

        # Load mask file
        if self.mask_files is not  None:
            mask_file = self.mask_files[index]
            msk, _ = raster_to_ndarray(
                                        mask_file,
                                        width=self.width,
                                        height=self.height,
                                        resolution=None,
                                        band_indices=self.mask_bands
                                        )
            sample["mask"] = msk

        # apply transforms
        if self.transform_function is None:
            self.transform_function = ToDoubleTensor()

        sample = self.transform_function(**sample)

        if self.get_sample_info:
            sample["filename"] = os.path.basename(image_file)
            sample["affine"] = affine_to_ndarray(meta["transform"])

        return sample


class PatchDetectionDataset(Dataset):

    def __init__(
        self,
        job,
        resolution,
        width,
        height,
        transform=None,
        image_bands=None
        ):

        self.job = job
        self.job.keep_only_todo_list()
        self.width = width
        self.height = height
        self.resolution = resolution
        self.transform_function = transform
        self.image_bands = image_bands

    def __len__(self):
        return len(self.job)

    def __getitem__(self, index):
        # load image file
        image_file = self.job.get_cell_at(index, "img_file")
        img, meta = raster_to_ndarray(
                                       image_file,
                                       width=self.width,
                                       height=self.height,
                                       resolution=self.resolution,
                                       band_indices=self.image_bands
                                       )
        img = img_as_float(img)  # pixels are normalized to [0, 1]
        to_tensor = ToPatchTensor()
        affine = meta["transform"]
        LOGGER.debug(affine)
        sample = {"image": img, "index": np.asarray([index]), "affine": affine_to_ndarray(affine)}
        sample = to_tensor(**sample)
        return sample


