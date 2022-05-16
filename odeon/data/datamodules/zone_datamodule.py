import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from pytorch_lightning import LightningDataModule
from rasterio.warp import aligned_target
from shapely import wkt
from torch.utils.data import DataLoader, Subset

from odeon import LOGGER
from odeon.commons.exception import ErrorCodes, OdeonError
from odeon.commons.rasterio import RIODatasetCollection, get_number_of_band
from odeon.data.datamodules.job import ZoneDetectionJob, ZoneDetectionJobNoDalle
from odeon.data.datasets import ZoneDetectionDataset

RANDOM_SEED = 42
BATCH_SIZE = 5
NUM_WORKERS = 4
PERCENTAGE_VAL = 0.3
OUTPUT_TYPE = "unint8"


class ZoneDataModule(LightningDataModule):
    def __init__(
        self,
        output_path: str,
        zone: Dict[str, Any],
        num_classes: int,
        img_size_pixel: Union[int, Tuple[int], List[int]],
        transforms: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        batch_size: Optional[int] = BATCH_SIZE,
        num_workers: Optional[int] = NUM_WORKERS,
        percentage_val: Optional[float] = PERCENTAGE_VAL,
        pin_memory: Optional[bool] = True,
        deterministic: Optional[bool] = False,
        get_sample_info: Optional[bool] = False,
        resolution: Optional[Union[int, Tuple[int], List[int]]] = None,
        output_type: Optional[str] = OUTPUT_TYPE,
        drop_last: Optional[bool] = False,
        subset: Optional[bool] = False,
    ):

        super().__init__()

        self.output_path = output_path
        self.num_classes = num_classes

        # Extraction of the value contained in zone.
        self.dict_of_raster = zone["sources"]
        self.extent = zone["extent"]
        self.tile_factor = zone["tile_factor"]
        self.margin = zone["margin_zone"]
        self.out_dalle_size = (
            zone["out_dalle_size"] if ("out_dalle_size" in zone.keys()) else None
        )
        self.dem = zone["dem"]

        self.transforms = transforms

        self.img_size_pixel = img_size_pixel
        self.width = (
            self.img_size_pixel[0] * self.tile_factor if width is None else width
        )
        self.height = (
            self.img_size_pixel[1] * self.tile_factor if height is None else height
        )
        self.output_size = [self.width, self.height]

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

        self.resolution = self.get_resolution(resolution)
        self.output_type = output_type
        self.batch_size = batch_size
        self.pred_dataset = None

        # Variables for zone detection
        self.meta = None
        self.job = None
        self.n_channel = None
        self.meta = None
        self.meta_output = None
        self.dst = None
        self.test_dataset = None

    def prepare_data(self):
        self.dst = rasterio.open(next(iter(self.dict_of_raster.values()))["path"])
        self.job = self.create_detection_job()
        self.job.save_job()
        self.n_channel = get_number_of_band(self.dict_of_raster, self.dem)
        self.meta = self.init_meta()

    def setup(self, stage=None):
        if not self.test_dataset:
            self.dataset = ZoneDetectionDataset(
                job=self.job,
                dict_of_raster=self.dict_of_raster,
                output_type=self.output_type,
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

        self.rio_ds_collection = RIODatasetCollection()
        self.meta = self.get_meta(next(iter(self.dict_of_raster.values()))["path"])
        self.meta["driver"] = "GTiff"
        self.meta["dtype"] = "uint8"
        self.meta["count"] = self.num_classes
        self.meta["transform"], _, _ = aligned_target(
            self.meta["transform"],
            self.meta["width"],
            self.meta["height"],
            self.resolution,
        )
        self.meta_output = self.meta.copy()
        if self.out_dalle_size is None:
            self.meta_output["width"] = self.img_size_pixel[0] * self.tile_factor - (
                2 * self.margin
            )
            self.meta_output["height"] = self.img_size_pixel[1] * self.tile_factor - (
                2 * self.margin
            )
        else:
            self.meta_output["width"] = math.ceil(
                self.out_dalle_size / self.resolution[0]
            )
            self.meta_output["height"] = math.ceil(
                self.out_dalle_size / self.resolution[1]
            )

    def get_meta(self, file):
        with rasterio.open(file) as src:
            return src.meta.copy()

    def get_resolution(self, resolution: float) -> Tuple[float]:
        output_resolution = []
        if isinstance(resolution, float):
            output_resolution = [resolution, resolution]
        elif isinstance(resolution, (tuple, list, np.ndarray)):
            output_resolution = resolution
        else:
            LOGGER.error(
                "ERROR: resolution parameter should be a float or a list/tuple of float"
            )
            raise OdeonError(
                ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                "ERROR: resolution parameter is not correct.",
            )
        return output_resolution
