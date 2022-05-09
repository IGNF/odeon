import os

# from rasterio.plot import reshape_as_raster
import numpy as np
import rasterio
from torch.utils.data import Dataset

from odeon import LOGGER
from odeon.commons.exception import ErrorCodes, OdeonError


class MetricsDataset(Dataset):
    def __init__(
        self,
        mask_files,
        pred_files,
        nbr_class,
        type_classifier,
        mask_bands=None,
        pred_bands=None,
        width=None,
        height=None,
    ):
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
        bands_unselected = [
            array[:, :, i]
            for i in list(set(np.arange(array.shape[-1])) - set(select_bands))
        ]
        bands_selected = np.stack(bands_selected, axis=-1)
        if bands_unselected:
            bands_unselected = np.stack(bands_unselected, axis=-1)
            bands_unselected = np.amax(bands_unselected, axis=-1).reshape(
                array.shape[0], array.shape[1], 1
            )
            bands_selected = np.concatenate([bands_selected, bands_unselected], axis=-1)
        return bands_selected

    def read_raster(self, path_raster, bands=None):
        with rasterio.open(path_raster) as raster:
            img = raster.read().swapaxes(0, 2).swapaxes(0, 1).astype(np.float32)
            if bands is None:
                return img if self.type_classifier == "multiclass" else img[:, :, 0]
            else:
                return (
                    self.select_bands(img, bands)
                    if self.type_classifier == "multiclass"
                    else img[:, :, bands]
                )

    def __getitem__(self, index):
        mask_file, pred_file = self.mask_files[index], self.pred_files[index]
        msk = self.read_raster(mask_file, self.mask_bands)
        pred = self.read_raster(pred_file, self.pred_bands)
        if not os.path.basename(mask_file) == os.path.basename(pred_file):
            LOGGER.error(
                "ERROR: %s is not present in masks files and in prediction files",
                os.path.basename(msk),
            )
            raise OdeonError(
                ErrorCodes.INVALID_DATASET_PATH,
                "The input parameter type classifier is incorrect.",
            )
        sample = {"mask": msk, "pred": pred, "name_file": mask_file}
        return sample
