import os
from torch.utils.data import Dataset
from skimage.util import img_as_float
import rasterio
# from rasterio.plot import reshape_as_raster
import numpy as np
from odeon import LOGGER
from odeon.commons.image import raster_to_ndarray, CollectionDatasetReader
from odeon.data.transforms.utils import ToDoubleTensor, ToPatchTensor, ToWindowTensor
from odeon.commons.rasterio import affine_to_ndarray
from odeon.commons.folder_manager import create_folder
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.data.datasets.patch import PatchDetectionDataset


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
            img = CollectionDatasetReader.get_stacked_window_collection(self.dict_of_raster,
                                                                        bounds,
                                                                        self.width,
                                                                        self.height,
                                                                        self.resolution,
                                                                        self.dem)
            to_tensor = ToWindowTensor()
            sample = {"image": img, "index": np.asarray([index])}
            sample = to_tensor(**sample)
            return sample

        except rasterio._err.CPLE_BaseError as error:
            LOGGER.warning(f"CPLE error {error}")
