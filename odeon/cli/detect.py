import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar

from odeon import LOGGER
from odeon.callbacks import PatchPredictionWriter, ZonePredictionWriter
from odeon.commons.core import BaseTool
from odeon.commons.exception import ErrorCodes, OdeonError
from odeon.commons.guard import dirs_exist, files_exist
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.data.datamodules import SegDataModule, ZoneDataModule
from odeon.modules.seg_module import SegmentationTask

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_detection")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)
ACCELERATOR = "gpu"
BATCH_SIZE = 5
NUM_WORKERS = 4
THRESHOLD = 0.5
DEFAULT_OUTPUT_TYPE = "uint8"
PROGRESS = 1
RANDOM_SEED = 42
NUM_PROCESSES = 1
NUM_NODES = 1


class DetectCLI(BaseTool):
    def __init__(
        self,
        verbosity: bool,
        # Model
        model_name: str,
        file_name: str,
        # Image
        img_size_pixel: Union[int, Tuple[int], List[int]],
        resolution: Union[int, Tuple[int], List[int]],
        # Output_param
        output_path: str,
        output_type: Optional[str] = DEFAULT_OUTPUT_TYPE,
        class_labels: Optional[List[str]] = None,
        sparse_mode: Optional[bool] = None,
        threshold: Optional[float] = THRESHOLD,
        # Detect_param
        batch_size: Optional[int] = BATCH_SIZE,
        device: Optional[Union[str, List[int], List[str]]] = None,
        accelerator: Optional[str] = ACCELERATOR,
        num_nodes: Optional[int] = NUM_NODES,
        num_processes: Optional[int] = NUM_PROCESSES,
        num_workers: Optional[int] = NUM_WORKERS,
        deterministic: Optional[bool] = False,
        strategy: Optional[str] = None,
        testing: Optional[bool] = False,
        get_metrics: Optional[bool] = True,
        progress: Optional[float] = PROGRESS,
        dataset: Optional[Dict[str, Any]] = None,  # Dataset
        zone: Optional[Dict[str, Any]] = None,  # Zone
    ):

        self.verbosity = verbosity

        # Model
        self.model_name = model_name
        self.model_filename = file_name

        # Image
        self.img_size_pixel = self.get_img_size_pixel(img_size_pixel)
        self.resolution = resolution

        # Output_param
        self.output_folder = output_path
        self.output_type = output_type
        self.class_labels = class_labels
        self.sparse_mode = sparse_mode
        self.threshold = threshold
        self.path_detections = os.path.join(self.output_folder, "detections")

        # Detect_param
        self.batch_size = batch_size
        self.device = device
        self.accelerator = accelerator
        self.num_nodes = num_nodes
        self.num_processes = num_processes
        self.num_workers = num_workers
        self.strategy = strategy
        self.testing = testing
        self.get_metrics = get_metrics
        self.progress_rate = progress

        self.df = None
        self.detector = None
        self.deterministic = deterministic
        self.zone = zone
        self.dataset = dataset

        # Main components used for the detection
        self.mode = None
        self.init_params = None
        self.data_module = None
        self.seg_module = None
        self.callbacks = None
        self.enable_progress_bar = None

    def __call__(self):
        try:

            self.check()

            self.configure()

            STD_OUT_LOGGER.info(
                f"Detection : \n"
                f"detection type: {self.mode} \n"
                f"device: {self.device} \n"
                f"model: {self.model_name} \n"
                f"model file: {self.model_filename} \n"
                f"number of classes: {self.data_module.num_classes} \n"
                f"batch size: {self.batch_size} \n"
                f"image size pixel: {self.img_size_pixel} \n"
                f"resolution: {self.resolution} \n"
                f"output type: {self.output_type}"
            )
            self.trainer.predict(
                model=self.seg_module,
                datamodule=self.data_module,
                ckpt_path=self.model_filename,
            )
        except OdeonError as error:
            raise OdeonError(
                ErrorCodes.ERR_DETECTION_ERROR,
                "ERROR: Something went wrong during the detection",
                stack_trace=error,
            )

    def check(self):
        try:
            files_to_check = [self.model_filename]
            files_exist(files_to_check)
            dirs_exist([self.output_folder])
        except OdeonError as error:
            raise OdeonError(
                ErrorCodes.ERR_DETECTION_ERROR,
                "something went wrong during detection configuration",
                stack_trace=error,
            )

    def configure(self):

        self.init_params = self.get_init_params()

        self.data_module = self.configure_datamodule(
            dataset=self.dataset, zone=self.zone
        )
        self.seg_module = SegmentationTask(**self.init_params)

        self.trainer = self.configure_trainer()

    def get_init_params(self):
        checkpoint = torch.load(self.model_filename)
        return checkpoint["hyper_parameters"]

    def setup(self):

        # Setup of random seed if the execution of the detection should be deterministic
        if self.deterministic is True:
            self.random_seed = RANDOM_SEED
            seed_everything(self.random_seed, workers=True)

        else:
            self.random_seed = None

    def configure_datamodule(self, dataset, zone):
        if dataset is not None and zone is not None:
            LOGGER.error(
                "ERROR: To use the detection tool you must fill in either the zone\
                part or the dataset part of the config file"
            )
            raise OdeonError(
                ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                "The configuration of the input JSON file is incorrect.",
            )

        elif zone is not None:
            self.mode = "zone"
            data_module = ZoneDataModule(
                output_path=self.output_folder,
                zone=zone,
                num_classes=self.init_params["num_classes"],
                img_size_pixel=self.img_size_pixel,
                transforms=None,
                width=None,
                height=None,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                deterministic=False,
                resolution=self.resolution,
                output_type=self.output_type,
                get_sample_info=True,
                subset=self.testing,
            )
        else:
            self.mode = "dataset"

            def _get(dataset: dict, key: str):

                if key in dataset.keys():
                    if key == "normalization_weights":
                        return {"test": dataset[key]}
                    else:
                        return dataset[key]
                else:
                    return None

            data_module = SegDataModule(
                test_file=_get(dataset, "path"),
                image_bands=_get(dataset, "image_bands"),
                mask_bands=_get(dataset, "mask_bands"),
                class_labels=self.class_labels,
                data_augmentation=None,
                data_stats=_get(dataset, "normalization_weights"),
                width=_get(dataset, "width"),
                height=_get(dataset, "height"),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                deterministic=False,
                resolution=self.resolution,
                get_sample_info=True,
                subset=self.testing,
            )
        return data_module

    def configure_callbacks(self):
        """
        Callbacks definition.
        """
        callbacks = []

        # Writer definition
        if self.mode == "dataset":
            custom_pred_writer = PatchPredictionWriter(
                output_dir=self.path_detections,
                output_type=self.output_type,
                write_interval="batch",
                img_size_pixel=self.img_size_pixel,
            )
        else:
            # self.mode == "zone"
            custom_pred_writer = ZonePredictionWriter(
                output_dir=self.path_detections,
                write_interval="batch",
            )

        callbacks = [custom_pred_writer]

        if self.progress_rate <= 0:
            self.enable_progress_bar = False
        else:
            progress_bar = TQDMProgressBar(refresh_rate=self.progress_rate)
            callbacks.append(progress_bar)
            self.enable_progress_bar = True

        return callbacks

    def configure_trainer(self):

        self.callbacks = self.configure_callbacks()

        return Trainer(
            devices=self.device,
            accelerator=self.accelerator,
            callbacks=self.callbacks,
            deterministic=self.deterministic,
            max_epochs=-1,
            strategy=self.strategy,
            num_nodes=self.num_nodes,
            num_processes=self.num_processes,
            enable_progress_bar=self.enable_progress_bar,
        )

    def get_img_size_pixel(self, get_img_size_pixel: int) -> Tuple[float]:
        output_img_size_pixel = []
        if isinstance(get_img_size_pixel, int):
            output_img_size_pixel = [get_img_size_pixel, get_img_size_pixel]
        elif isinstance(get_img_size_pixel, (tuple, list, np.ndarray)):
            output_img_size_pixel = get_img_size_pixel
        else:
            LOGGER.error(
                "ERROR: image_size_pixel parameter should be a int or a list/tuple of int"
            )
            raise OdeonError(
                ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                "ERROR: image_size_pixel parameter is not correct.",
            )
        return output_img_size_pixel
