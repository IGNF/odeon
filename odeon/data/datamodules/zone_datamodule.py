import os

import geopandas as gpd
import rasterio
from pytorch_lightning import LightningDataModule
from rasterio.warp import aligned_target
from shapely import wkt
from torch.utils.data import DataLoader, Subset

from odeon.commons.exception import ErrorCodes, OdeonError
from odeon.commons.rasterio import get_number_of_band
from odeon.data.datamodules.job import (ZoneDetectionJob,
                                        ZoneDetectionJobNoDalle)
from odeon.data.datasets import ZoneDetectionDataset

RANDOM_SEED = 42
BATCH_SIZE = 5
NUM_WORKERS = 4
PERCENTAGE_VAL = 0.3


class ZoneDataModule(LightningDataModule):
    def __init__(
        self,
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
        subset=False,
    ):

        super().__init__()

        self.output_path = output_path
        self.n_classes = None

        # Extraction of the value contained in zone.
        self.dict_of_raster = zone["sources"]
        self.extent = zone["extent"]
        self.tile_factor = zone["tile_factor"]
        self.margin = zone["margin"]
        self.out_dalle_size = (
            zone["out_dalle_size"] if "out_dalle_size" in zone.keys() else None
        )
        self.dem = zone["dem"]
        self.img_size_pixel = img_size_pixel
        self.output_size = self.img_size_pixel * self.tile_factor
        self.width = self.img_size_pixel * self.tile_factor
        self.margin = margin
        self.transforms = transforms

        self.width = self.img_size_pixel * self.tile_factor if width is None else width
        self.height = (
            self.img_size_pixel * self.tile_factor if height is None else height
        )

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

        self.resolution = resolution
        self.batch_size = batch_size
        self.pred_dataset = None

        # Variable for zone detection
        self.zone_detection_job = None
        self.n_channel = None
        self.meta = None

    def prepare_data(self):
        self.zone_detection_job = self.create_detection_job()
        self.zone_detection_job.save_job()
        self.n_channel = get_number_of_band(self.dict_of_raster, self.dem)
        self.meta = self.init_meta()

    def setup(self):
        if not self.test_dataset:
            self.dataset = ZoneDetectionDataset(
                job=self.zone_detection_job,
                dict_of_raster=self.dict_of_raster,
                meta=self.meta,
                dem=self.dem,
                height=self.height,
                width=self.width,
                resolution=self.resolution,
            )
        if self.subset is True:
            self.test_dataset = Subset(self.test_dataset, range(0, 10))

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=self.drop_last,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=self.drop_last,
        )

    def create_gdf(self):
        if os.path.exists(self.extent):
            if os.path.isfile(self.extent):
                # Target if a file
                gdf_zone = gpd.GeoDataFrame.from_file(self.extent)
            else:
                # Target is a directory containing multiple files .shp, .shx, .dbf, .prj ...
                with rasterio.open(
                    next(iter(self.dict_of_raster.values()))["path"]
                ) as src:
                    crs = src.crs

                gdf_zone = gpd.GeoDataFrame(
                    data=[{"id": 1, "geometry": wkt.loads(self.extent)}],
                    geometry="geometry",
                    crs=crs,
                )
            return gdf_zone
        else:
            raise OdeonError(
                ErrorCodes.ERR_FILE_NOT_EXIST, f"{self.extent} doesn't exists"
            )

    def create_detection_job(self):
        gdf_zone = self.create_gdf()
        df, _ = ZoneDetectionJob.build_job(
            gdf=gdf_zone,
            output_size=self.output_size,
            resolution=self.resolution,
            overlap=self.margin,
            out_dalle_size=self.out_dalle_size,
        )
        if self.out_dalle_size is not None:
            zone_detection_job = ZoneDetectionJob(df, self.output_path)
        else:
            zone_detection_job = ZoneDetectionJobNoDalle(df, self.output_path)
        return zone_detection_job

    def init_meta(self):
        self.meta = self.get_meta(self.zone_detection_job.get_cell_at(0, "img_file"))
        self.meta["driver"] = "GTiff"
        self.meta["dtype"] = "uint8"
        self.meta["count"] = self.n_classes
        self.meta["transform"], _, _ = aligned_target(
            self.meta["transform"],
            self.meta["width"],
            self.meta["height"],
            self.resolution,
        )
        self.meta["width"] = self.img_size_pixel
        self.meta["height"] = self.img_size_pixel

    def get_meta(self, file):
        with rasterio.open(file) as src:
            return src.meta.copy()
