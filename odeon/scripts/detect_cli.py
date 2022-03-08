import os
import yaml
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import (
    Trainer,
    seed_everything
)
from odeon.callbacks.utils_callbacks import (
    CustomPredictionWriter,
    HistorySaver
)
from odeon.commons.core import BaseTool
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.guard import dirs_exist, files_exist
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.modules.datamodule import SegDataModule
from odeon.modules.seg_module import SegmentationTask
from odeon.nn.transforms import Compose, ToDoubleTensor

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_detection")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)
ACCELERATOR = "gpu"
BATCH_SIZE = 5
NUM_WORKERS = 4
THRESHOLD = 0.5
DEFAULT_OUTPUT_TYPE="uint8"
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
        hparams_file=None,
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
        # Dataset
        dataset=None,
        # Zone
        zone=None,
        ):

        self.verbosity = verbosity

        # Model
        self.model_name = model_name
        self.model_filename = file_name
        self.hparams_file = hparams_file

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

        self.model_ext = os.path.splitext(self.model_filename)[-1]

        # implement TTA?
        self.transforms = {"test": Compose([ToDoubleTensor()])}

        if zone is not None:
            self.mode = "zone"
            self.zone = zone
            print("In zone part, will be implemented soon...")

        else:
            self.mode = "dataset"
            self.dataset =dataset

            self.data_module = SegDataModule(test_file=self.dataset["path"],
                                             image_bands=self.dataset["image_bands"],
                                             mask_bands=self.dataset["mask_bands"],
                                             transforms=self.transforms,
                                             width=None,
                                             height=None,
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             deterministic=False,
                                             resolution=self.resolution,
                                             get_sample_info=True,
                                             subset=self.testing)

        if class_labels is not None:
            if len(class_labels) == self.data_module.num_classes:
                self.class_labels = class_labels
            else:
                LOGGER.error('ERROR: parameter labels should have a number of values equal to the number of classes.')
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                    "The input parameter labels is incorrect.")
        else:
            self.class_labels = [f'class {i + 1}' for i in range(self.data_module.num_classes)]

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
            self.configure()
        except OdeonError as error:
            raise error

        except Exception as error:
            raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                             "something went wrong during detection configuration",
                             stack_trace=error)

    def __call__(self):
        try:

            predict_ckpt = None
            if self.model_ext == ".ckpt":
                predict_ckpt = self.model_filename

            self.trainer.predict(self.seg_module,
                                 datamodule=self.data_module,
                                 ckpt_path=predict_ckpt)

        except OdeonError as error:
            raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                                "ERROR: Something went wrong during the test step of the training",
                                stack_trace=error)

    def check(self):
        try:
            files_to_check = [self.model_filename]
            files_exist(files_to_check)
            dirs_exist([self.output_folder])
        except OdeonError as error:
            raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                             "something went wrong during detection configuration",
                             stack_trace=error)

    def configure(self):

        if self.model_ext == ".ckpt":
            self.init_params = torch.load(self.model_filename)["hyper_parameters"]
            self.seg_module = SegmentationTask(**self.init_params)

        elif self.model_ext == ".pth" and self.hparams_file is not None:
            with open(self.hparams_file, "r") as stream:
                try:
                    self.init_params = yaml.safe_load(stream)
                except yaml.YAMLError as error:
                    raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                                    "something went wrong during detection configuration",
                                    stack_trace=error)
            self.seg_module = SegmentationTask(**self.init_params)
            self.seg_module.setup()
            model_state_dict = torch.load(os.path.join(self.model_filename))
            self.seg_module.model.load_state_dict(state_dict=model_state_dict)
            LOGGER.info(f"Prediction with file :{self.model_filename}")

        else:
            LOGGER.error('ERROR: Detection tool work only with .ckpt and .pth files. For .pth you have to declare a hparams_file')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                "The input parameter file_name is incorrect.")

        # Loggers definition
        loggers = []
        detect_csv_logger = CSVLogger(save_dir=os.path.join(self.output_folder),
                                      name="detect_csv")
        loggers.append(detect_csv_logger)

        # Callbacks definition
        path_detections = os.path.join(self.output_folder, "detections")
        custom_pred_writer = CustomPredictionWriter(output_dir=path_detections,
                                                    output_type=self.output_type,
                                                    write_interval="batch")
        self.callbacks = [custom_pred_writer]

        if self.get_metrics:
            self.callbacks.append(HistorySaver())

        if self.progress_rate <= 0 :
            self.enable_progress_bar = False
        else :
            progress_bar = TQDMProgressBar(refresh_rate=self.progress_rate)
            self.callbacks.append(progress_bar)
            self.enable_progress_bar = True

        self.trainer = Trainer(devices=self.device,
                               accelerator=self.accelerator,
                               callbacks=self.callbacks,
                               logger=loggers,
                               deterministic=self.deterministic,
                               strategy=self.strategy,
                               num_nodes=self.num_nodes,
                               num_processes=self.num_processes,
                               enable_progress_bar=self.enable_progress_bar)
