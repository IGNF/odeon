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


class DetectCLI(BaseTool):

    def __init__(self,
                 verbosity,
                 img_size_pixel,
                 resolution,
                 model_name,
                 file_name,
                 n_classes,
                 batch_size,
                 use_gpu,
                 idx_gpu,
                 interruption_recovery,
                 mutual_exclusion,
                 output_path,
                 output_type,
                 sparse_mode,
                 threshold,
                 margin=None,
                 num_worker=None,
                 num_thread=None,
                 dataset=None,
                 zone=None,
                 device=None,
                 accelerator=ACCELERATOR,
                 num_nodes=1,
                 num_processes=None,
                 ):

        self.verbosity = verbosity
        self.img_size_pixel = img_size_pixel
        self.resolution = resolution if isinstance(resolution, list) else [resolution, resolution]
        self.margin = margin
        self.model_name = model_name
        self.file_name = file_name
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        STD_OUT_LOGGER.info(f"""CUDA available? {torch.cuda.is_available()}""")
        self.use_gpu = False if torch.cuda.is_available() is False else self.use_gpu
        self.idx_gpu = idx_gpu
        self.num_worker = num_worker
        self.num_thread = num_thread
        self.interruption_recovery = interruption_recovery
        self.output_path = output_path
        self.output_type = output_type
        self.sparse_mode = sparse_mode
        self.threshold = threshold
        self.df = None
        self.detector = None
        self.mutual_exclusion = mutual_exclusion

        if zone is not None:
            self.mode = "zone"
            self.zone = zone
        else:
            self.mode = "dataset"
            self.dataset = dataset
        STD_OUT_LOGGER.info(f"""detection type: {self.mode}

device: {"cuda" if self.use_gpu else "cpu"}
model: {self.model_name}
model file: {self.file_name}
number of classes: {self.n_classes}
batch size: {self.batch_size}
image size pixel: {self.img_size_pixel}
resolution: {self.resolution}
activation: {"softmax" if self.mutual_exclusion is True else "sigmoïd"}
output type: {self.output_type}""")
        if self.mode == "zone":

            STD_OUT_LOGGER.info(f"""overlap margin: {self.zone["margin_zone"]}
compute digital elevation model: {self.zone["dem"]}
tile factor: {self.zone["tile_factor"]}
            """)
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

            self.detector = ZoneDetector(dict_of_raster=dict_of_raster,
                                         extent=extent,
                                         tile_factor=tile_factor,
                                         margin_zone=margin_zone,
                                         job=zone_detection_job,
                                         output_path=self.output_path,
                                         model_name=self.model_name,
                                         file_name=self.file_name,
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
                                         dem=dem,
                                         out_dalle_size=out_dalle_size)

            LOGGER.debug(self.detector.__dict__)
            self.detector.configure()
