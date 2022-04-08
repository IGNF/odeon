import os
from torch.utils.data import Dataset
from skimage.util import img_as_float
import rasterio
# from rasterio.plot import reshape_as_raster
import numpy as np
from odeon.commons.image import raster_to_ndarray, CollectionDatasetReader
from odeon.nn.transforms import ToDoubleTensor, ToPatchTensor, ToWindowTensor
from odeon import LOGGER
from odeon.commons.rasterio import affine_to_ndarray
from odeon.commons.folder_manager import create_folder
from odeon.commons.exception import OdeonError, ErrorCodes


class MetricsDataset(Dataset):

    def __init__(self,
                 mask_files,
                 pred_files,
                 nbr_class,
                 type_classifier,
                 mask_bands=None,
                 pred_bands=None,
                 width=None,
                 height=None):
        self.mask_files = mask_files
        self.pred_files = pred_files
        self.nbr_class = nbr_class
        self.width = width
        self.height = height
        self.patch_size = width * height
        self.type_classifier = type_classifier
        self.mask_bands = mask_bands
        self.pred_bands = pred_bands

    def __len__(self):
        return len(self.mask_files)

    @staticmethod
    def select_bands(array, select_bands):
        """
        Function allowing to select bands in a mask/prediction array thanks to a list containing the indices of the
        bands you want to extract. The other unselected bands will be grouped into a single one, which will contain
        the largest value among them for a given pixel.

        Parameters
        ----------
        array : np.array
            Arrays on which we want to extract the bands.
        select_bands : list of int
            List containing the indices of the bands to extract.
        """
        bands_selected = [array[:, :, i] for i in select_bands]
        bands_unselected = [array[:, :, i] for i in list(set(np.arange(array.shape[-1])) - set(select_bands))]
        bands_selected = np.stack(bands_selected, axis=-1)
        if bands_unselected:
            bands_unselected = np.stack(bands_unselected, axis=-1)
            bands_unselected = np.amax(bands_unselected, axis=-1).reshape(array.shape[0], array.shape[1], 1)
            bands_selected = np.concatenate([bands_selected, bands_unselected], axis=-1)
        return bands_selected

    def read_raster(self, path_raster, bands=None):
        with rasterio.open(path_raster) as raster:
            img = raster.read().swapaxes(0, 2).swapaxes(0, 1).astype(np.float32)
            if bands is None:
                return img if self.type_classifier == 'multiclass' else img[:, :, 0]
            else:
                return self.select_bands(img, bands) if self.type_classifier == 'multiclass' else img[:, :, bands]

    def __getitem__(self, index):
        mask_file, pred_file = self.mask_files[index], self.pred_files[index]
        msk = self.read_raster(mask_file, self.mask_bands)
        pred = self.read_raster(pred_file, self.pred_bands)
        if not os.path.basename(mask_file) == os.path.basename(pred_file):
            LOGGER.error("ERROR: %s is not present in masks files and in prediction files", os.path.basename(msk))
            raise OdeonError(ErrorCodes.INVALID_DATASET_PATH,
                             "The input parameter type classifier is incorrect.")
        sample = {"mask": msk, "pred": pred, "name_file": mask_file}
        return sample


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
                 mask_bands=None, get_sample_info=False):

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

        # load mask file
        mask_file = self.mask_files[index]
        msk, _ = raster_to_ndarray(
                                    mask_file,
                                    width=self.width,
                                    height=self.height,
                                    resolution=None,
                                    band_indices=self.mask_bands
                                    )

        sample = {"image": img, "mask": msk}

        # apply transforms
        if self.transform_function is None:
            self.transform_function = ToDoubleTensor()

        sample = self.transform_function(**sample)

        if self.get_sample_info:
            sample["filename"] = os.path.basename(image_file)
            sample["affine"] = affine_to_ndarray(meta["transform"])

        return sample


class PatchDetectionDataset(Dataset):

    def __init__(self,
                 job,
                 resolution,
                 width,
                 height,
                 transform=None,
                 image_bands=None):

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

        # pixels are normalized to [0, 1]
        img = img_as_float(img)
        to_tensor = ToPatchTensor()
        affine = meta["transform"]
        LOGGER.debug(affine)
        sample = {"image": img, "index": np.asarray([index]), "affine": affine_to_ndarray(affine)}
        sample = to_tensor(**sample)
        return sample


class ZoneDetectionDataset(PatchDetectionDataset):

    def __init__(self,
                 job,
                 resolution,
                 width,
                 height,
                 dict_of_raster,
                 output_type,
                 meta,
                 transform=None,
                 dem=False,
                 gdal_options=None,
                 export_input=False,
                 export_path=None
                 ):

        super(ZoneDetectionDataset, self).__init__(job,
                                                   resolution,
                                                   width,
                                                   height,
                                                   transform)

        self.dict_of_raster = dict_of_raster
        self.gdal_options = gdal_options
        self.dem = dem
        self.export_input = export_input
        self.export_path = export_path
        self.meta = meta
        if self.export_path is not None:

            create_folder(self.export_path)

    def __enter__(self):

        for key in self.dict_of_raster.keys():

            if self.gdal_options is None:

                self.dict_of_raster[key]["connection"] = rasterio.open(self.dict_of_raster[key]["path"])
            else:

                self.dict_of_raster[key]["connection"] = rasterio.open(self.dict_of_raster[key]["path"],
                                                                       **self.gdal_options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        for key in self.dict_of_raster.keys():

            self.dict_of_raster[key]["connection"].close()

    def __len__(self):

        return len(self.job)

    def __getitem__(self, index):

        try:

            bounds = self.job.get_bounds_at(index)
            # LOGGER.info(image_file)
            # LOGGER.info(self.image_files)
            img = CollectionDatasetReader.get_stacked_window_collection(self.dict_of_raster,
                                                                        bounds,
                                                                        self.width,
                                                                        self.height,
                                                                        self.resolution,
                                                                        self.dem)

            to_tensor = ToWindowTensor()
            # affine = meta["transform"]
            # LOGGER.debug(affine)
            sample = {"image": img, "index": np.asarray([index])}
            sample = to_tensor(**sample)
            return sample

        except rasterio._err.CPLE_BaseError as error:

            LOGGER.warning(f"CPLE error {error}")
