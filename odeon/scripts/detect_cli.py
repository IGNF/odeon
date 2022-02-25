import os
import pandas as pd
import rasterio
import torch
import geopandas as gpd
from shapely import wkt
from odeon.commons.core import BaseTool
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.guard import dirs_exist, files_exist
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.commons.rasterio import get_number_of_band
from odeon.nn.detector import BaseDetector, PatchDetector, ZoneDetector
from odeon.nn.job import PatchJobDetection, ZoneDetectionJob, ZoneDetectionJobNoDalle

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_detection")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)
ACCELERATOR = "gpu"
BATCH_SIZE = 5
NUM_WORKERS = 4
THRESHOLD = 0.5


class DetectCLI(BaseTool):

    def __init__(
        self,
        model_name,
        model_filename,
        output_path,
        output_type="uint8",
        img_size_pixel=None,
        test_file=None,
        image_bands=None,
        mask_bands=None,
        class_labels=None,
        resolution=None,
        batch_size=BATCH_SIZE,
        device=None,
        accelerator=ACCELERATOR,
        num_nodes=1,
        num_processes=None,
        num_workers=NUM_WORKERS,
        strategy=None,
        name_exp_log=None,
        version_name=None,
        testing=False,
        threshold=THRESHOLD,
        margin=None,
        zone=None,
        sparse_mode=None,
        dataset=None,
        ):

        self.model_name = model_name
        self.model_filename = model_filename
        self.output_path = output_path
        self.output_type = output_type
        self.img_size_pixel = img_size_pixel
        self.test_file = test_file
        self.image_bands = image_bands
        self.mask_bands = mask_bands
        self.class_labels = class_labels
        self.resolution = resolution
        self.batch_size = batch_size
        self.device = device
        self.accelerator = accelerator
        self.num_nodes = num_nodes
        self.num_processes = num_processes
        self.num_workers = num_workers
        self.strategy = strategy
        self.version_name = version_name
        self.testing = testing
        self.threshold = threshold
        self.margin = margin
        self.sparse_mode = sparse_mode
        self.dataset = dataset
        self.img_size_pixel = img_size_pixel
        self.resolution = resolution
        self.margin = margin

        self.df = None
        self.detector = None

        if zone is not None:
            self.mode = "zone"
            self.zone = zone
        else:
            self.mode = "dataset"

        STD_OUT_LOGGER.info(
            f"Detection : \n" 
            f"detection type: {self.mode} \n"
            f"device: {self.device} \n"
            f"model: {self.model_name} \n"
            f"model file: {self.file_name} \n"
            f"number of classes: {self.data_module.num_classes} \n"
            f"batch size: {self.batch_size} \n"
            f"image size pixel: {self.img_size_pixel} \n"
            f"resolution: {self.resolution} \n"
            f"output type: {self.output_type}"
            )

        try:
            self.check()
            self.configure()
        except OdeonError as error:
            raise error

        except Exception as error:
            raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                             "something went wrong during detection configuration",
                             stack_trace=error)

    def __call__(self):
        self.detector.run()

    def check(self):
        try:
            files_exist([self.file_name])
            dirs_exist([self.output_path])
        except OdeonError as error:
            raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                             "something went wrong during detection configuration",
                             stack_trace=error)
        else:
            pass

    def configure(self):
        if self.mode == "dataset":
            if self.dataset["path"].endswith(".csv"):
                self.df = pd.read_csv(self.dataset["path"], usecols=[0], header=None, names=["img_file"])
            else:
                img_array = [f for f in os.listdir(self.dataset["path"]) if os.path.isfile(os.path.join(
                    self.dataset["path"], f))]
                self.df = pd.DataFrame(img_array, columns={"img_file": str})
            self.df["img_output_file"] = self.df.apply(lambda row: os.path.join(self.output_path,
                                                                                str(row["img_file"]).split("/")[-1]),
                                                       axis=1)
            self.df["job_done"] = False
            self.df["transform"] = object()
            if "image_bands" in self.dataset.keys():
                image_bands = self.dataset["image_bands"]
                n_channel = len(image_bands)
            else:
                with rasterio.open(self.df["img_file"].iloc[0]) as src:
                    n_channel = src.count
                    image_bands = range(1, n_channel + 1)

            patch_detection_job = PatchJobDetection(self.df, self.output_path, self.interruption_recovery)
            self.detector = PatchDetector(patch_detection_job,
                                          self.output_path,
                                          self.model_name,
                                          self.file_name,
                                          n_classes=self.n_classes,
                                          n_channel=n_channel,
                                          img_size_pixel=self.img_size_pixel,
                                          resolution=self.resolution,
                                          batch_size=self.batch_size,
                                          use_gpu=self.use_gpu,
                                          idx_gpu=self.idx_gpu,
                                          num_worker=self.num_worker,
                                          num_thread=self.num_thread,
                                          mutual_exclusion=self.mutual_exclusion,
                                          output_type=self.output_type,
                                          sparse_mode=self.sparse_mode,
                                          threshold=self.threshold,
                                          verbosity=self.verbosity,
                                          image_bands=image_bands)
            self.detector.configure()
        else:
            dict_of_raster = self.zone["sources"]
            with rasterio.open(next(iter(dict_of_raster.values()))["path"]) as src:
                crs = src.crs
                LOGGER.debug(crs)
            if os.path.isfile(self.zone["extent"]):
                gdf_zone = gpd.GeoDataFrame.from_file(self.zone["extent"])
            else:
                gdf_zone = gpd.GeoDataFrame([{"id": 1, "geometry": wkt.loads(self.zone["extent"])}],
                                            geometry="geometry",
                                            crs=crs)
            LOGGER.debug(gdf_zone)
            extent = self.zone["extent"]
            tile_factor = self.zone["tile_factor"]
            margin_zone = self.zone["margin_zone"]
            output_size = self.img_size_pixel * tile_factor
            out_dalle_size = self.zone["out_dalle_size"] if "out_dalle_size" in self.zone.keys() else None
            LOGGER.debug(f"output_size {out_dalle_size}")

            self.df, _ = ZoneDetectionJob.build_job(gdf=gdf_zone,
                                                    output_size=output_size,
                                                    resolution=self.resolution,
                                                    overlap=self.zone["margin_zone"],
                                                    out_dalle_size=out_dalle_size)

            LOGGER.debug(len(self.df))

            if out_dalle_size is not None:
                zone_detection_job = ZoneDetectionJob(self.df,
                                                      self.output_path,
                                                      self.interruption_recovery)
            else:
                zone_detection_job = ZoneDetectionJobNoDalle(self.df,
                                                             self.output_path,
                                                             self.interruption_recovery)
            zone_detection_job.save_job()
            dem = self.zone["dem"]
            n_channel = get_number_of_band(dict_of_raster, dem)
            LOGGER.debug(f"number of channel input: {n_channel}")



            LOGGER.debug(self.detector.__dict__)
            self.detector.configure()
