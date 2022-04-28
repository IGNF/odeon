import os
import csv
import numpy as np
import rasterio
import geopandas as gpd
from shapely import wkt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
from odeon import LOGGER
from odeon.data.datasets.patch_dataset import PatchDataset
from odeon.data.datamodules.job import (
    ZoneDetectionJob,
    ZoneDetectionJobNoDalle
)
from odeon.commons.guard import check_files, check_raster_bands
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.guard import file_exist
from odeon.commons.rasterio import get_number_of_band

RANDOM_SEED = 42
BATCH_SIZE = 5
NUM_WORKERS = 4
PERCENTAGE_VAL = 0.3


class ZoneDataModule(LightningDataModule):

    def __init__(self,
                 output_path, 
                 zone,
                 img_size_pixel,
                 margin=None,
                 transforms=None,
                 width=None,
                 height=None,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS,
                 percentage_val=PERCENTAGE_VAL,
                 pin_memory=True,
                 deterministic=False,
                 get_sample_info=False,
                 resolution=None,
                 drop_last=False,
                 subset=False):

        super().__init__()
        
        self.output_path = output_path

        # Extraction of the value contained in zone.
        self.dict_of_raster = zone["sources"]
        self.extent = zone["extent"]
        self.tile_factor = zone["tile_factor"]
        self.margin = zone["margin"]
        self.out_dalle_size = zone["out_dalle_size"] if "out_dalle_size" in zone.keys() else None
        self.dem = zone["dem"]
        self.img_size_pixel = img_size_pixel
        self.output_size = self.img_size_pixel * self.tile_factor
        
        self.margin = margin
        self.transforms = transforms

        self.width = width
        self.height = height

        self.num_workers = num_workers
        self.percentage_val = percentage_val

        self.pin_memory = pin_memory
        self.get_sample_info = get_sample_info
        self.drop_last = drop_last 
        self.subset = subset

        if deterministic:
            self.random_seed = None
            self.shuffle = False
        else:
            self.random_seed = RANDOM_SEED
            self.shuffle = True
        self.test_image_files, self.test_mask_files = None, None
        self.resolution = resolution
        self.batch_size = batch_size
        self.test_dataset, self.pred_dataset = None, None

    def prepare_data(self):
        self.zone_detection_job = self.create_detection_job()
        self.zone_detection_job.save_job()
        self.n_channel = get_number_of_band(self.dict_of_raster, self.dem)

    def setup(self, stage=None):         
        if not self.pred_dataset:
            self.test_dataset = PatchDataset(image_files=self.test_image_files,
                                                 mask_files=self.test_mask_files,
                                                 transform=self.transforms['test'],
                                                 image_bands=self.image_bands,
                                                 mask_bands=self.mask_bands,
                                                 width=self.width,
                                                 height=self.height,
                                                 get_sample_info=self.get_sample_info)
        if self.subset is True:
            self.test_dataset = Subset(self.test_dataset, range(0, 10))

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False,
                          drop_last=self.drop_last)

    def predict_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False,
                          drop_last=self.drop_last)

    def create_gdf(self):
        if os.path.exists(self.extent):
            if os.path.isfile(self.extent):
                # Target if a file 
                gdf_zone = gpd.GeoDataFrame.from_file(self.extent)

            else:
                # Target is a directory containing multiple files .shp, .shx, .dbf, .prj ...
                with rasterio.open(next(iter(self.dict_of_raster.values()))["path"]) as src:
                    crs = src.crs

                gdf_zone = gpd.GeoDataFrame(data=[{"id": 1, "geometry": wkt.loads(self.extent)}],
                                            geometry="geometry",
                                            crs=crs)

            return gdf_zone

        else:
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                    f"{self.extent} doesn't exists")

    def create_detection_job(self):
        gdf_zone = self.create_gdf()
        df, _ = ZoneDetectionJob.build_job(gdf=gdf_zone,
                                                output_size=self.output_size,
                                                resolution=self.resolution,
                                                overlap=self.margin,
                                                out_dalle_size=self.out_dalle_size)
        if self.out_dalle_size is not None:
            zone_detection_job = ZoneDetectionJob(df, self.output_path)
        else:
            zone_detection_job = ZoneDetectionJobNoDalle(df, self.output_path)
        
        return zone_detection_job
