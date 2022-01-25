"""
module of Detection jobs
"""
import os
import math
import multiprocessing
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import rasterio
from rasterio.features import geometry_window
from rasterio.windows import transform
from rasterio.plot import reshape_as_raster
from rasterio.warp import aligned_target
from odeon.nn.datasets import PatchDetectionDataset, ZoneDetectionDataset
from odeon.nn.models import build_model, model_list
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon import LOGGER
from odeon.commons.rasterio import ndarray_to_affine, RIODatasetCollection
from odeon.commons.image import TypeConverter, substract_margin
from odeon.commons.shape import create_polygon_from_bounds
from odeon.commons.folder_manager import create_folder
NB_PROCESSOR = multiprocessing.cpu_count()


class BaseDetector:

    def __init__(self,
                 job,
                 output_path,
                 model_name,
                 model_path,
                 n_classes,
                 n_channel,
                 img_size_pixel=256,
                 resolution=[0.2, 0.2],
                 batch_size=16,
                 use_gpu=True,
                 idx_gpu=None,
                 num_worker=None,
                 num_thread=None,
                 mutual_exlusion=True,
                 output_type="uint8",
                 sparse_mode=False,
                 threshold=0.5,
                 verbosity=False
                 ):

        self.verbosity = verbosity
        self.img_size_pixel = img_size_pixel
        self.resolution = resolution
        self.model_name = model_name
        self.model_path = model_path
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.idx_gpu = idx_gpu
        self.num_worker = num_worker
        self.num_thread = num_thread
        self.mutual_exclusion = mutual_exlusion
        self.output_path = output_path
        self.output_type = output_type
        self.sparse_mode = sparse_mode
        self.threshold = threshold
        self.job = job
        self.data_loader = None
        self.dataset = None
        self.device = self.get_device()
        self.model = BaseDetector.load_model(self.model_name,
                                             self.model_path,
                                             self.n_channel,
                                             self.n_classes,
                                             self.device)

    def configure(self):

        pass

    def get_device(self):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            if self.idx_gpu is None and self.use_gpu is True:
                device = "cuda:0"
            elif self.idx_gpu is not None:
                list_gpus = list(range(torch.cuda.device_count()))
                if self.idx_gpu in list_gpus:
                    device = "cuda:" + str(self.idx_gpu)
                else:
                    LOGGER.error(f"ERROR: Input GPU reference doesn't exist in available GPU devices: {list_gpus}")
                    raise OdeonError(message=f"Input GPU reference cuda:{self.idx_gpu} doesn't exist",
                                     error_code=ErrorCodes.ERR_DETECTION_ERROR)
        else:
            device = 'cpu'
        LOGGER.info(f"INFO: Job launched on device: {device}")
        return device

    @staticmethod
    def load_model(model_name, model_path, n_channel, n_classes, device='cpu'):
        if model_name not in model_list:
            raise OdeonError(message=f"the model name {model_name} does not exist",
                             error_code=ErrorCodes.ERR_MODEL_ERROR)
        model = build_model(model_name, n_channel, n_classes)
        model.to(device)
        state_dict = torch.load(model_path,
                                map_location=torch.device(device))
        model.load_state_dict(state_dict=state_dict)
        model.eval()  # drop dropout and batchnorm for inference mode
        return model

    def run(self):

        if len(self.job) > 0:
            try:
                for samples in tqdm(self.data_loader):
                    predictions = self.detect(samples["image"])
                    LOGGER.debug(predictions)
                    indices = samples["index"].cpu().numpy()
                    affines = samples["affine"].cpu().numpy()
                    self.save(predictions, indices, affines)

            except KeyboardInterrupt as error:
                LOGGER.warning("the job has been prematurely interrupted")
                raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                                 "something went wrong during detection",
                                 stack_trace=error)

            except Exception as error:
                raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                                 "something went wrong during detection",
                                 stack_trace=error)
            finally:
                self.job.save_job()
                LOGGER.debug("the detection job has been saved")

        else:

            LOGGER.warning(f""""job has no work to do, maybe
            your input directory or csv file is empty, or you may have set the
             interruption_recovery at true while the output directory
             {self.output_path} contain a csv job file of previous work""")

    def detect(self, images):
        """

        Parameters
        ----------
        images

        Returns
        -------

        """

        if self.use_gpu:

            images = images.cuda()

        with torch.no_grad():

            logits = self.model(images)

        # predictions

        if self.model.n_classes == 1:

            predictions = torch.sigmoid(logits)

        else:

            if self.mutual_exclusion is True:

                predictions = F.softmax(logits, dim=1)

            else:

                predictions = torch.sigmoid(logits)

        predictions = predictions.cpu().numpy()

        return predictions

    def save(self, predictions, indices, affines):

        pass


class PatchDetector(BaseDetector):

    """

    """
    def __init__(self,
                 job,
                 output_path,
                 model_name,
                 file_name,
                 n_classes,
                 n_channel,
                 img_size_pixel=256,
                 resolution=[0.2, 0.2],
                 batch_size=16,
                 use_gpu=True,
                 idx_gpu=None,
                 num_worker=None,
                 num_thread=None,
                 mutual_exclusion=True,
                 output_type="uint8",
                 sparse_mode=False,
                 threshold=0.5,
                 verbosity=False,
                 image_bands=None
                 ):

        super(PatchDetector, self).__init__(
             job,
             output_path,
             model_name,
             file_name,
             n_classes,
             n_channel,
             img_size_pixel,
             resolution,
             batch_size,
             use_gpu,
             idx_gpu,
             num_worker,
             num_thread,
             mutual_exclusion,
             output_type,
             sparse_mode,
             threshold,
             verbosity)

        self.meta = None
        self.image_bands = image_bands
        self.resolution = resolution
        self.job = job
        self.num_thread = num_thread
        self.num_worker = num_worker
        self.gdal_options = {"compress": "LZW",
                             "tiled": True,
                             "blockxsize": self.img_size_pixel,
                             "blockysize": self.img_size_pixel,
                             "SPARSE_MODE": self.sparse_mode}

        if self.output_type == "bit":

            self.gdal_options["bit"] = 1

    def configure(self):

        LOGGER.debug(len(self.job))
        self.meta = self.get_meta(self.job.get_cell_at(0, "img_file"))
        self.meta["driver"] = "GTiff"
        self.meta["dtype"] = "uint8" if self.output_type in ["uint8", "bit"] else "float32"
        self.meta["count"] = self.n_classes
        self.meta["transform"], _, _ = aligned_target(self.meta["transform"],
                                                      self.meta["width"],
                                                      self.meta["height"],
                                                      self.resolution)
        self.meta["width"] = self.img_size_pixel
        self.meta["height"] = self.img_size_pixel
        self.dataset = PatchDetectionDataset(self.job,
                                             height=self.img_size_pixel,
                                             width=self.img_size_pixel,
                                             image_bands=self.image_bands,
                                             resolution=self.resolution)

        self.num_worker = NB_PROCESSOR if self.num_worker is None else self.num_worker
        self.num_thread = NB_PROCESSOR if self.num_thread is None else self.num_thread
        torch.set_num_threads(self.num_thread)
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=min(self.batch_size, len(self.job)),
                                      num_workers=self.num_worker,
                                      pin_memory=True
                                      )

    @classmethod
    def get_meta(cls, file):

        with rasterio.open(file) as src:

            return src.meta.copy()

    def save(self, predictions, indices, affines):

        for prediction, index, affine in zip(predictions, indices, affines):

            LOGGER.debug(index)
            output_file = self.job.get_cell_at(index[0], "img_output_file")
            self.job.set_cell_at(index[0], "transform", affine)
            self.meta["transform"] = ndarray_to_affine(affine)

            with rasterio.open(output_file, "w", **self.meta, **self.gdal_options) as src:

                converter = TypeConverter()
                prediction = converter.from_type("float32").to_type(self.output_type).convert(prediction,
                                                                                              threshold=self.threshold)
                src.write(prediction)
                self.job.set_cell_at(index[0], "job_done", True)


class ZoneDetector(PatchDetector):
    """

    """

    def __init__(self,
                 dict_of_raster,
                 extent,
                 tile_factor,
                 margin_zone,
                 job,
                 output_path,
                 model_name,
                 file_name,
                 n_classes,
                 n_channel,
                 img_size_pixel=256,
                 resolution=[0.2, 0.2],
                 batch_size=16,
                 use_gpu=True,
                 idx_gpu=None,
                 num_worker=None,
                 num_thread=None,
                 mutual_exclusion=True,
                 output_type="uint8",
                 sparse_mode=False,
                 threshold=0.5,
                 verbosity=False,
                 dem=False,
                 out_dalle_size=None
                 ):

        super(ZoneDetector, self).__init__(
            job,
            output_path,
            model_name,
            file_name,
            n_classes,
            n_channel,
            img_size_pixel,
            resolution,
            batch_size,
            use_gpu,
            idx_gpu,
            num_worker,
            num_thread,
            mutual_exclusion,
            output_type,
            sparse_mode,
            threshold,
            verbosity)

        self.dict_of_raster = dict_of_raster
        self.extent = extent
        self.tile_factor = tile_factor
        self.margin_zone = margin_zone
        self.dem = dem
        self.output_write = os.path.join(self.output_path, "result")
        create_folder(self.output_write)
        self.gdal_options["BIGTIFF"] = "YES"
        self.dst = None
        self.meta_output = None
        self.out_dalle_size = out_dalle_size
        self.rio_ds_collection = None
        LOGGER.debug(out_dalle_size)

    def configure(self):

        LOGGER.debug(len(self.job))
        self.rio_ds_collection = RIODatasetCollection()

        self.dst = rasterio.open(next(iter(self.dict_of_raster.values()))["path"])
        self.meta = self.dst.meta.copy()
        self.meta["driver"] = "GTiff"
        self.meta["dtype"] = "uint8" if self.output_type in ["uint8", "bit"] else "float32"
        self.meta["count"] = self.n_classes
        self.meta["transform"], _, _ = aligned_target(self.meta["transform"],
                                                      self.meta["width"],
                                                      self.meta["height"],
                                                      self.resolution)
        self.meta_output = self.meta.copy()
        if self.out_dalle_size is None:

            self.meta_output["height"] = self.img_size_pixel * self.tile_factor - (2 * self.margin_zone)
            self.meta_output["width"] = self.meta_output["height"]

        else:

            self.meta_output["height"] = math.ceil(self.out_dalle_size / self.resolution[1])
            self.meta_output["width"] = math.ceil(self.out_dalle_size / self.resolution[0])

        self.num_worker = 0 if self.num_worker is None else self.num_worker
        self.num_thread = NB_PROCESSOR if self.num_thread is None else self.num_thread
        torch.set_num_threads(self.num_thread)

    def save(self, predictions, indices):

        for prediction, index in zip(predictions, indices):

            prediction = prediction.transpose((1, 2, 0)).copy()
            # LOGGER.info(prediction.shape)
            prediction = substract_margin(prediction, self.margin_zone, self.margin_zone)
            prediction = reshape_as_raster(prediction)
            converter = TypeConverter()
            prediction = converter.from_type("float32").to_type(self.output_type).convert(prediction,
                                                                                          threshold=self.threshold)

            output_id = self.job.get_cell_at(index[0], "output_id")
            LOGGER.debug(output_id)
            name = str(output_id) + ".tif"
            output_file = os.path.join(self.output_write, name)

            if self.out_dalle_size is not None and self.rio_ds_collection.collection_has_key(output_id):

                out = self.rio_ds_collection.get_rio_dataset(output_id)
                # LOGGER.info(f"{str(output_id)}in ds collection")

            else:
                # LOGGER.info(f"{str(output_id)} not in ds collection")
                left = self.job.get_cell_at(index[0], "left_o")
                bottom = self.job.get_cell_at(index[0], "bottom_o")
                right = self.job.get_cell_at(index[0], "right_o")
                top = self.job.get_cell_at(index[0], "top_o")

                geometry = create_polygon_from_bounds(left, right, bottom, top)
                window = geometry_window(
                                         self.dst,
                                         [geometry],
                                         pixel_precision=6).round_shape(op='ceil', pixel_precision=4)

                # window = self.dst.window(left, right, bottom, top)
                self.meta_output["transform"] = transform(window, self.dst.transform)
                out = rasterio.open(output_file, 'w+', **self.meta_output, **self.gdal_options)
                LOGGER.debug(out.bounds)
                LOGGER.debug(self.dst.bounds)
                # exit(0)
                self.rio_ds_collection.add_rio_dataset(output_id, out)

            LOGGER.debug(out.meta)

            left = self.job.get_cell_at(index[0], "left")
            bottom = self.job.get_cell_at(index[0], "bottom")
            right = self.job.get_cell_at(index[0], "right")
            top = self.job.get_cell_at(index[0], "top")
            geometry = create_polygon_from_bounds(left, right, bottom, top)
            LOGGER.debug(geometry)
            window = geometry_window(
                                    out,
                                    [geometry],
                                    pixel_precision=6).round_shape(op='ceil', pixel_precision=4)
            LOGGER.debug(window)
            indices = [i for i in range(1, self.n_classes + 1)]

            out.write_band([i for i in range(1, self.n_classes + 1)], prediction, window=window)

            self.job.set_cell_at(index[0], "job_done", 1)

            if self.out_dalle_size is not None and self.job.job_finished_for_output_id(output_id):

                self.rio_ds_collection.delete_key(output_id)
                self.job.mark_dalle_job_as_done(output_id)
                self.job.save_job()
                # LOGGER.info(f"{str(output_id)} removed from ds collection")

            if self.out_dalle_size is None:

                out.close()

            # self.write_job.save_job()

    def run(self):

        if len(self.job) > 0:

            try:

                self.dataset = ZoneDetectionDataset(job=self.job,
                                                    dict_of_raster=self.dict_of_raster,
                                                    output_type=self.output_type,
                                                    meta=self.meta,
                                                    dem=self.dem,
                                                    height=self.img_size_pixel * self.tile_factor,
                                                    width=self.img_size_pixel * self.tile_factor,
                                                    resolution=self.resolution)

                with self.dataset as dataset:

                    LOGGER.debug(f"length: {len(dataset.job)}")

                    self.data_loader = DataLoader(dataset,
                                                  batch_size=min(self.batch_size, len(self.job)),
                                                  num_workers=self.num_worker,
                                                  pin_memory=True
                                                  )

                    for samples in tqdm(self.data_loader):

                        # LOGGER.debug(samples)
                        predictions = self.detect(samples["image"])
                        # LOGGER.debug(predictions)
                        indices = samples["index"].cpu().numpy()
                        LOGGER.debug(indices)
                        self.save(predictions, indices)

            except KeyboardInterrupt as error:

                LOGGER.warning("the job has been prematurely interrupted")
                raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                                 "something went wrong during detection",
                                 stack_trace=error)

            except rasterio._err.CPLE_BaseError as error:

                LOGGER.warning(f"CPLE error {error}")

            except Exception as error:

                raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                                 "something went wrong during detection",
                                 stack_trace=error)

            finally:

                self.job.save_job()

                if self.dst is not None:

                    self.dst.close()

                # LOGGER.info("the detection job has been saved")

        else:

            LOGGER.warning(f""""job has no work to do, maybe
            your input directory or csv file is empty, or you may have set the
            interruption_recovery at true while the output directory
            {self.output_path} contain a job file of previous work completed
            (all the work has been done)""")
