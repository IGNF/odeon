import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from odeon.callbacks.history import HistorySaver
from odeon.callbacks.writer import PatchPredictionWriter
from odeon.commons.core import BaseTool
from odeon.commons.exception import ErrorCodes, OdeonError
from odeon.commons.guard import dirs_exist, files_exist
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.data.datamodules.patch_datamodule import SegDataModule
from odeon.data.datamodules.zone_datamodule import ZoneDataModule
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
        verbosity,
        # Model
        model_name,
        file_name,
        # Image
        img_size_pixel,
        resolution,
        # Output_param
        output_path,
        output_type=DEFAULT_OUTPUT_TYPE,
        class_labels=None,
        sparse_mode=None,
        threshold=THRESHOLD,
        # Detect_param
        batch_size=BATCH_SIZE,
        device=None,
        accelerator=ACCELERATOR,
        num_nodes=NUM_NODES,
        num_processes=NUM_PROCESSES,
        num_workers=NUM_WORKERS,
        deterministic=False,
        strategy=None,
        testing=False,
        get_metrics=True,
        progress=PROGRESS,
        dataset=None,  # Dataset
        zone=None,  # Zone
    ):

        self.verbosity = verbosity

        # Model
        self.model_name = model_name
        self.model_filename = file_name

        # Image
        self.img_size_pixel = img_size_pixel
        self.resolution = resolution

        # Output_param
        self.output_folder = output_path
        self.output_type = output_type
        self.class_labels = class_labels
        self.sparse_mode = sparse_mode
        self.threshold = threshold

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

        if deterministic is True:
            self.random_seed = RANDOM_SEED
            seed_everything(self.random_seed, workers=True)
            self.deterministic = False
        else:
            self.random_seed = None
            self.deterministic = False

        if zone is not None:
            self.mode = "zone"
            self.zone = zone
            self.data_module = ZoneDataModule(
                zone=self.zone,
                img_size_pixel=self.img_size_pixel,
                transforms=None,
                width=None,
                height=None,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                deterministic=False,
                resolution=self.resolution,
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

            self.dataset = dataset
            self.data_module = SegDataModule(
                test_file=_get(self.dataset, "path"),
                image_bands=_get(self.dataset, "image_bands"),
                mask_bands=_get(self.dataset, "mask_bands"),
                class_labels=self.class_labels,
                data_augmentation=None,
                data_stats=_get(self.dataset, "normalization_weights"),
                width=_get(self.dataset, "width"),
                height=_get(self.dataset, "height"),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                deterministic=False,
                resolution=self.resolution,
                get_sample_info=True,
                subset=self.testing,
            )

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

        try:
            self.check()
        except OdeonError as error:
            raise error

        except Exception as error:
            raise OdeonError(
                ErrorCodes.ERR_DETECTION_ERROR,
                "something went wrong during detection configuration",
                stack_trace=error,
            )

    def __call__(self):
        try:
            self.configure()
            self.trainer.predict(
                model=self.seg_module,
                datamodule=self.data_module,
                ckpt_path=self.model_filename
            )
        except OdeonError as error:
            raise OdeonError(
                ErrorCodes.ERR_DETECTION_ERROR,
                "ERROR: Something went wrong during the test step of the training",
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
        self.init_params = torch.load(self.model_filename)["hyper_parameters"]
        self.seg_module = SegmentationTask(**self.init_params)

        # Loggers definition
        loggers = []
        detect_csv_logger = CSVLogger(
            save_dir=os.path.join(self.output_folder), name="detect_csv"
        )
        loggers.append(detect_csv_logger)

        # Callbacks definition

        # Writer definition
        if self.mode == "dataset":
            path_detections = os.path.join(self.output_folder, "detections")
            custom_pred_writer = PatchPredictionWriter(
                output_dir=path_detections,
                output_type=self.output_type,
                write_interval="batch",
                img_size_pixel=self.img_size_pixel,
            )
        else:
            # Zone mode
            # path_detections = os.path.join(self.output_folder, "detections")
            # custom_pred_writer = ZonePredictionWriter(
            #     output_dir=path_detections,
            #     output_type=self.output_type,
            #     write_interval="batch",
            #     img_size_pixel=self.img_size_pixel,
            # )
            pass

        self.callbacks = [custom_pred_writer]

        if self.get_metrics:
            self.callbacks.append(HistorySaver())

        if self.progress_rate <= 0:
            self.enable_progress_bar = False
        else:
            progress_bar = TQDMProgressBar(refresh_rate=self.progress_rate)
            self.callbacks.append(progress_bar)
            self.enable_progress_bar = True

        self.trainer = Trainer(
            devices=self.device,
            accelerator=self.accelerator,
            callbacks=self.callbacks,
            logger=loggers,
            deterministic=self.deterministic,
            max_epochs=-1,
            strategy=self.strategy,
            num_nodes=self.num_nodes,
            num_processes=self.num_processes,
            enable_progress_bar=self.enable_progress_bar,
        )
