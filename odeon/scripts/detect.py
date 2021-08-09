"""
Entry point of the Detect CLI tool.
This module aims to perform a detection based on an extent or a collection of extent
with an Odeon model and a dictionary of raster input
"""
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


class DetectionTool(BaseTool):
    """Main entry point of detection tool

    Implements
    ----------
    BaseTool : object
        the abstract class for implmenting a CLI tool
    """
    def __init__(self,
                 verbosity,
                 img_size_pixel,
                 resolution,
                 model_name,
                 file_name,
                 n_classes,
                 batch_size,
                 use_gpu,
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
                 zone=None
                 ):
        """[summary]

        Parameters
        ----------
        verbosity : boolean
            verbosity of logger
        img_size_pixel : int
            image size of output in pixel
        resolution : Union(float, list(float, float))
            output resolution in x and y
        model_name : str
            name of te model as declared in the nn.models.build_model function
        file_name : str
            file of trained model with only weights parameters.
        n_classes : int
            The number of class learned by the model
        batch_size : int
            the size of the batch in the dataloader
        use_gpu : boolead
            use a GPU or not
        interruption_recovery : boolean
            store and restart from where the detection has been
            if an interruption has been encountered
        mutual_exclusion : boolean
            In multiclass model you can use softmax if True or
            Sigmoïd if False
        output_path : str
            output path of the detection
        output_type : str
            the output type, one of int8, float32 or bit
        sparse_mode : boolean
            if set to True, will only write the annotated pixels on disk.
            If can save a lot of space.
        threshold : float beetwen 0 and 1
            threshold used in the case of an output in bit (0/1)
        margin : int, optional
            a margin to use to make an overlaping detection.
            It can improve your prediction by ignoring the bordered pixels
            of the prediction, by default None
        num_worker : int, optional
            Number of worker used by the dataloader.
            Stable with the prediction by dataset but not with
            a prediction by zone, by default None (0 extra worker)
        num_thread : int, optional
            Number of thread used during the prediction.
            Useful when you infer on CPU, by default None
        dataset : dict, optional
            the description of an inference by dataset, by default None
        zone : dict, optional
            the description of an inference by zone, by default None

        Raises
        ------

        OdeonError
            ERR_DETECTION_ERROR, if something goes wrong during the prediction
        """
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
        self.num_worker = num_worker
        self.num_thread = num_thread
        self.interruption_recovery = interruption_recovery
        self.output_path = output_path
        self.output_type = output_type
        self.sparse_mode = sparse_mode
        self.threshold = threshold
        self.df: pd.DataFrame = None
        self.detector: BaseDetector = None
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
        """Call the Detector implemented (by zone, or by dataset)
        """
        # LOGGER.debug(self.__dict__)
        self.detector.run()

    def check(self):
        """Check configuration
        if there is an anomaly in the input parameters.

        Raises
        ------
        OdeonError
            ERR_DETECTION_ERROR, f something wrong has been detected in parameters
        """

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
        """Configuraiton of the Detector class used to make the
        detection
        """

        if self.mode == "dataset":

            if self.dataset["path"].endswith(".csv"):

                self.df = pd.read_csv(self.dataset["path"], usecols=[0], header=None, names=["img_file"])
                # Intéressant à rajouter pour les métriques?
                # self.df = pd.read_csv(self.dataset["path"], usecols=[1], header=None, names=["msk_file"])
            else:

                img_array = [f for f in os.listdir(self.dataset["path"]) if os.path.isfile(os.path.join(
                    self.dataset["path"], f))]
                self.df = pd.DataFrame(img_array, columns={"img_file": str})

            self.df["img_output_file"] = self.df.apply(lambda row: os.path.join(self.output_path,
                                                                                str(row["img_file"]).split("/")[-1]),
                                                       axis=1)

            self.df["job_done"] = False
            self.df["transform"] = object()
            # self.df = self.df.head(250)

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

            """"
            Build Job
            """

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

            # self.df = self.df.sample(n=100, random_state=1).reset_index()
            # self.df.to_file("/home/dlsupport/data/33/ground_truth/2018/learning_zones/test_zone_1.shp")

            if out_dalle_size is not None:

                zone_detection_job = ZoneDetectionJob(self.df,
                                                      self.output_path,
                                                      self.interruption_recovery)
            else:

                zone_detection_job = ZoneDetectionJobNoDalle(self.df,
                                                             self.output_path,
                                                             self.interruption_recovery)
            zone_detection_job.save_job()
            # write_job = WriteJob(df_write, self.output_path, self.interruption_recovery, file_name="write_job.shp")
            # write_job.save_job()
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
