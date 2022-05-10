import os
from datetime import date
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from odeon import LOGGER
from odeon.callbacks import (
    ContinueTraining,
    ExoticCheckPoint,
    GraphAdder,
    HistogramAdder,
    HistorySaver,
    HParamsAdder,
    LightningCheckpoint,
    MetricsAdder,
    PatchPredictionWriter,
    PredictionsAdder,
)
from odeon.commons.core import BaseTool
from odeon.commons.exception import ErrorCodes, OdeonError
from odeon.commons.guard import dirs_exist, file_exist
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.data.datamodules import SegDataModule
from odeon.loggers import JSONLogger
from odeon.models.base import MODEL_LIST
from odeon.modules import SegmentationTask

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_training")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)
RANDOM_SEED = 42
PERCENTAGE_VAL = 0.3
VAL_CHECK_INTERVAL = 1.0
BATCH_SIZE = 3
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
NUM_WORKERS = 4
NUM_CKPT_SAVED = 3
MODEL_OUT_EXT = ".ckpt"
ACCELERATOR = "gpu"
PROGRESS = 1
NUM_NODES = 1
EARLY_STOPPING_CONFIG = {
    "patience": 30,
    "monitor": "val_loss",
    "mode": "min",
    "min_delta": 0.00,
}


class TrainCLI(BaseTool):
    """
    Main entry point of training tool

    Implements
    ----------
    BaseTool : object
        the abstract class for implementing a CLI tool
    """

    def __init__(
        self,
        verbosity: bool,
        model_name: str,
        output_folder: str,
        train_file: str,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        percentage_val: Optional[float] = PERCENTAGE_VAL,
        image_bands: Optional[List[int]] = None,
        mask_bands: Optional[List[int]] = None,
        class_labels: Optional[List[str]] = None,
        resolution: Optional[Union[float, List[float]]] = None,
        model_filename: Optional[str] = None,
        model_out_ext: Optional[str] = None,
        normalization_weights: Optional[Dict[str, List[float]]] = None,
        epochs: Optional[int] = NUM_EPOCHS,
        batch_size: Optional[int] = BATCH_SIZE,
        save_history: Optional[bool] = True,
        continue_training: Optional[bool] = False,
        loss: Optional[str] = "ce",
        class_imbalance: Optional[List[float]] = None,
        optimizer_config: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        early_stopping: Optional[Dict] = EARLY_STOPPING_CONFIG,
        data_augmentation: Optional[Dict[str, Union[str, List[str]]]] = None,
        device: Optional[Union[str, List[int], List[str]]] = None,
        accelerator: Optional[str] = ACCELERATOR,
        num_nodes: Optional[int] = NUM_NODES,
        num_processes: Optional[int] = None,
        random_seed: Optional[int] = RANDOM_SEED,
        deterministic: Optional[bool] = False,
        lr: Optional[float] = LEARNING_RATE,
        num_workers: Optional[int] = NUM_WORKERS,
        strategy: Optional[str] = None,
        val_check_interval: Optional[float] = VAL_CHECK_INTERVAL,
        name_exp_log: Optional[str] = None,
        version_name: Optional[str] = None,
        use_tensorboard: Optional[bool] = True,
        use_wandb: Optional[bool] = False,
        log_learning_rate: Optional[bool] = False,
        save_top_k: Optional[int] = NUM_CKPT_SAVED,
        get_prediction: Optional[bool] = False,
        prediction_output_type: Optional[str] = "uint8",
        testing: Optional[bool] = False,
        progress: Optional[float] = PROGRESS,
    ) -> None:
        """
            Init function of the CLI used for the train tool.

        Parameters
        ----------
        verbosity: bool
            Verbose level of the outputs.
        model_name: str
            Name of the architecture model to use to do the training. Available models are "unet",
            "lightunet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet150", "deeplab".
        output_folder: str
            Path where the output files will be stored. Outputs files will be at least the training
            model (.ckpt or .pth). In addition, one could logs training metrics in tensorboard, and/or
            a single file dictionary, and/or in Weight & Biases (wandb). If a csv file for the test
            split is provided, metrics will be computed or predictions could be also done with trained
            model (using the one with the best validation metrics at an epoch).
        train_file: str
            Path to the csv file containing the data in the train split which should be used for the training.
        val_file: str, optional
            Path to the csv file containing the data for in the validation split which should be used for the training.
            If None, the data for validation split will be obtained by splitting the data in the train file using
            the percentage_val parameter, by default None.
        test_file: str, optional
            Path to the csv file containing the data for in the test split which should be used for the training.
            If provided, metrics will be computed on those data or predictions will be made if the, by default None.
        percentage_val: float, optional
            If the validation file (val_file) is not provided this parameter will be used to split the training
            data into a training split and a validation split. For example, if percentage_val = 0.3, then 0.7
            of data from train_file will be used in the train split and 0.3 will be used for the validation split,
            by default 0.3.
        image_bands: List[int], optional
            List of the channel of the image to use for the training. Only specified bands of input images will be used
            in training. If not provided, all the bands of the image will be selected, by default None.
        mask_bands: List[int], optional
            List of the band of the mask (classes) to use for the training. Only specified bands of input masks will be
            used in training. If this parameter is not provided, all the bands of the mask will be selected,
            by default None.
        class_labels: List[str], optional
            List of the labels for each class used for the training. Should have the same number of value as mask_band
            parameter. If None, labels will be "class 1", "class 2" ... to "class n" for every class selected,
            by default None.
        resolution: Union[float, Tuple[float], List[float]], optional
            Resolution of the image in the dataset. Could be define for the whole dataset or for each split.
        model_out_ext: str, optional
            Define the output type of the model which could be ".ckpt" or ".pth". If not provided the output trained
            model will be of type ".ckpt", by default None.
        model_filename: str, optional
            Name for the output trained model. The name of the output depend on the extension type of the output model
            (could be define in model_out_ext). If model type is ".ckpt" there will multiple output trained models and
            each will contains the basename of the input model_filename. If model type is ".pth" there will only one
            trained model with name defined in the model_filename parameter, by default None.
        normalization_weights: Dict[str, List[float]], optional
            Dict of stats (mean, std) for each split (train/val/test) to do a mean std normalization: (X - mean) / std.
            Those stats are in range 0-1 for each image band used in training. If not provided, the normalization will
            be by scaling values in range 0-255 to 0-1, by default None.
        epochs: int, optional
            Number of epochs for which the training will be performed, by default 1000.
        batch_size: int, optional
            Number of samples used in a mini-batch, default 3.
        save_history: bool, optional
            Parameter to save the metrics of the training for the validation phase for each epoch (could be also done
            for test phase if test_file is provided) in JSON file, by default True.
        continue_training: bool, optional
            Parameter to resume a training from a former trained model. A training could be resume from a checkpoint
            file or from a .pth file. If the parameter is set to true, the model to resume will be search at the path:
            output_folder/model_filename. The type of the model file will be automatically detected and if the file
            is of type ".pth" other files (optimizer and history) could be passed (by putting thoses files at the
            same location output_folder) to resume more precisely a training, by default False.
        loss: str, optional
            Loss function used for the training. Available parameters are "ce": cross-entropy loss, "bce":binary cross
            entropy loss, "focal": focal loss, "combo": combo loss (a mix between "bce", "focal"), by default "ce".
        class_imbalance: List[float], optional
            A list of weights for each class. Weights will be used to balance the cost function.
            Usable only when loss is set to "ce", default None.
        optimizer_config: dict, optional
            A dictionary containing parameters for the optimizer. Available optimizer are: "adam", "radam", "adamax",
            "sgd", "rmsprop". The parameters of each optimizer are configurable by entering the name of the parameter
            as a key and the associated value in the configuration dictionary (you can look at the pytorch
            documentation of those classes at https://pytorch.org/docs/stable/optim.html), by default None.
        scheduler_config: dict, optional
            A dictionary containing parameters for the scheduler. Available scheduler are: "reducelronplateau",
            "cycliclr", "cosineannealinglr", "cosineannealingwarmrestarts". The parameters of each scheduler are
            configurable by entering the name of the parameter as a key and the associated value in the configuration
            dictionary (you can look at the pytorch documentation of those classes at
            https://pytorch.org/docs/stable/optim.html), by default None.
        data_augmentation: Dict[str, List[str]], optional
            Dictionary defining for each split the data augmentation transformation that one want to apply. Available
            data augmentation are rotation90 and radiometry or both. Now data augmentation can only be applied to the
            training set and not on the validation or test set. If nothing is defined the transformation on the data
            will be only a normalization and casting of array to tensor, by default None.
        device: Union[List[int], str, int], optional
            Number(s) or id(s) of device(s) to use. Will be mapped to either gpus, tpu_cores, num_processes or ipus,
            based on the accelerator type, by default None.
        accelerator: str, optional
            Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “auto”) as well as custom
            accelerator instances, by default None.
        num_nodes: int, optional
            Number of GPU nodes for distributed training, by default 1.
        num_processes: int, optional
            Number of processes for distributed training with accelerator="cpu", by default 1.
        random_seed: int, optional
            Value used to initialize the random number generator. The random number generator needs a number to start
            with (a seed value), to be able to generate a random number, by default 42.
        deterministic: bool, optional
            If True, sets whether PyTorch operations must use deterministic algorithms, by default False.
        lr: float, optional
            Learning rate for the training, by default 0.001.
        num_workers: int, optional
            How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process,
            by default 4.
        strategy: str, optional
            Supports different training strategies with aliases as well custom strategies. In order to do multi-gpus
            training use the strategy ddp, by default None.
        val_check_interval: float, optional
            How often to check the validation set. Pass a float in the range [0.0, 1.0] to check after a fraction of
            the training epoch. Pass an int to check after a fixed number of training batches, by default 1.
        name_exp_log: str, optional
            Name of the experience of the training (for example unet_ocsge). The folder will be inside the output
            folder if it doesn't exists it will be created. If this parameter is not provided the name of the
            experience will be the day the training have been launched(ex: 2022-06-17), by default None.
        version_name: str, optional
            Name of the version of the training (for example version_0) which will be inside the experience folder.
            This system of version allows the user to have easily multiple versions of one experience, those versions
            could be the same or with little tweaks in the configuration files. If this parameter is not provided the
            name of the version will be the time at which the training have been launched (ex: 17-08-53),
            by default None.
        use_tensorboard: bool, optional
            If set to by default True. The metrics of the training will be stored with tensorboard loggers. For a
            training there will be a logger for the train and validation (and test if test_file is provided) phases.
            Each logger will contains metrics, model graph, distributions and histograms of model's weights, and also
            images with their related masks anf predictions, by defaut True.
        use_wandb: bool, optional
            If set to True, the metrics will be logged using Weight and Biases (wandb) logger. This WANDB logger allows
            to save the metrics and also the code used for the training. The output files will be stored as local files
            and also will be synchronized in the web application of WANDB (https://wandb.ai), by default False.
        log_learning_rate: bool, optional
            If set to True, the value of the learning rate (and its momentum) will be logged, by default False.
        early_stopping: dict, optional
            This parameter will set a condition (for example a threshold on a monitored metric) which will stop the
            training if this condition is no more fulfilled. The default condition will be that the validation loss
            ("monitor") must go down ("mode") with a delta of at least 0.00 ("min_delta") before the end of a
            duration of 30 epochs ("patience"). This example will have for config,
            by default {"patience": 30, "monitor": "val_loss", "mode": "min", "min_delta": 0.00}.
        save_top_k: int, optional
            Number of checkpoints saved by training (for a monitored metric). The checkpoints will be selected
            according to a monitored metrics, here we watch two metrics: the validation loss (we keep the k models
            with the lowest val_loss) and the mIoU (macro IoU/mean of IoU per class) on the validation set (we keep
            the k models with the highest val_miou), so if k=3 we will save 6 checkpoints. This parameter is only
            used if output trained model is of type ".ckpt", by default 3.
        get_prediction: bool, optional
            Parameter could be only used if the test_file is provided. The predictions will be made with the mode with
            the best val_loss model and the predictions will be sotred in a predicitons folder inside the experience
            folder, by default False.
        prediction_output_type: str, optional
            Type of the output predictions. This parameter will be used only if the parameter "get_prediction"
            is set to True, by default "uint8".
        testing: bool, optional
            Have to be used for modifications testing or debugging. If set to True, only a subset of the data will
            be used in the training pipeline, by default False.
        progress: float, optional
            Determines at which rate (in number of batches) the progress bars get updated. Set it to 0 to disable
            the display. By default, the Trainer uses this implementation of the progress bar and sets the refresh
            rate to the value provided to the progress_bar_refresh_rate argument in the Trainer, by default 1.
        """
        self.verbosity = verbosity
        # Parameters for outputs
        self.output_folder = output_folder
        self.name_exp_log = name_exp_log
        self.version_name = version_name
        self.model_filename = model_filename
        self.model_out_ext = model_out_ext
        # Dict of desired data augmentation
        self.data_augmentation = data_augmentation
        # Stats for each split (mean and std) in order to do normalization
        self.normalization_weights = normalization_weights

        # Datamodule
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.percentage_val = percentage_val
        self.batch_size = batch_size
        self.image_bands = image_bands
        self.mask_bands = mask_bands
        self.class_labels = class_labels
        self.resolution = resolution
        self.num_workers = num_workers
        self.testing = testing

        # Segmentation Module
        self.loss_name = loss
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.learning_rate = lr
        self.class_imbalance = class_imbalance
        self.model_name = model_name

        # Loggers
        self.use_tensorboard = use_tensorboard
        self.log_learning_rate = log_learning_rate

        # Callbacks
        self.get_prediction = get_prediction
        self.prediction_output_type = prediction_output_type
        self.save_history = save_history
        self.continue_training = continue_training
        self.use_wandb = use_wandb
        self.early_stopping = early_stopping
        self.save_top_k = save_top_k
        self.progress_rate = progress
        self.enable_progress_bar = None

        # Training parameters
        self.epochs = epochs
        self.deterministic = deterministic
        self.random_seed = random_seed
        self.val_check_interval = val_check_interval

        # Parameters for device definition
        self.accelerator = accelerator
        self.num_nodes = num_nodes
        self.num_processes = num_processes
        self.device = device
        self.strategy = strategy

        # Definition training main modules
        self.callbacks = None
        self.resume_checkpoint = None
        self.transforms = None
        self.data_module = None
        self.seg_module = None
        self.loggers = None
        self.callbacks = None
        self.trainer = None

        self.setup()  # Define output paths (exp, logs, version name)
        self.check()  # Check model name and if output folders exist.

    def configure(self):

        self.data_module = SegDataModule(
            train_file=self.train_file,
            val_file=self.val_file,
            test_file=self.test_file,
            image_bands=self.image_bands,
            mask_bands=self.mask_bands,
            data_augmentation=self.data_augmentation,
            data_stats=self.normalization_weights,
            width=None,
            height=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            percentage_val=self.percentage_val,
            pin_memory=True,
            deterministic=self.deterministic,
            get_sample_info=self.get_prediction,
            resolution=self.resolution,
            subset=self.testing,
            random_seed=self.random_seed,
        )

        self.seg_module = SegmentationTask(
            model_name=self.model_name,
            num_classes=self.data_module.num_classes,
            num_channels=self.data_module.num_channels,
            class_labels=self.data_module.class_labels,
            criterion_name=self.loss_name,
            learning_rate=self.learning_rate,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            loss_classes_weights=self.class_imbalance,
            deterministic=self.deterministic,
        )

        self.loggers = self.configure_loggers()

        self.callbacks = self.configure_callbacks()

        self.trainer = Trainer(
            val_check_interval=self.val_check_interval,
            devices=self.device,
            accelerator=self.accelerator,
            callbacks=self.callbacks,
            max_epochs=self.epochs,
            logger=self.loggers,
            deterministic=self.deterministic,
            strategy=self.strategy,
            num_nodes=self.num_nodes,
            num_processes=self.num_processes,
            enable_progress_bar=self.enable_progress_bar,
        )

    def __call__(self):
        """
        Call the Trainer
        """
        self.configure()

        self.data_module.setup(stage="fit")

        STD_OUT_LOGGER.info(
            f"Training : \n"
            f"device: {self.device} \n"
            f"model: {self.model_name} \n"
            f"model file: {self.model_filename} \n"
            f"number of classes: {self.data_module.num_classes} \n"
            f"number of samples: {len(self.data_module.train_dataset) + len(self.data_module.val_dataset)}"
            f"(train: {len(self.data_module.train_dataset)}, val: {len(self.data_module.val_dataset)})"
        )

        try:
            self.trainer.fit(
                self.seg_module,
                datamodule=self.data_module,
                ckpt_path=self.resume_checkpoint,
            )

        except OdeonError as error:
            raise OdeonError(
                ErrorCodes.ERR_TRAINING_ERROR,
                "ERROR: Something went wrong during the fit step of the training",
                stack_trace=error,
            )

        if self.test_file is not None:
            try:
                best_val_loss_ckpt_path = None
                if self.model_out_ext == ".ckpt":
                    ckpt_val_loss_folder = os.path.join(
                        self.output_folder,
                        self.name_exp_log,
                        "odeon_val_loss_ckpt",
                        self.version_name,
                    )
                    best_val_loss_ckpt_path = self.get_path_best_ckpt(
                        ckpt_folder=ckpt_val_loss_folder, monitor="val_loss", mode="min"
                    )

                elif self.model_out_ext == ".pth":
                    # Load model weights into the model of the seg module
                    best_model_state_dict = torch.load(
                        os.path.join(self.output_folder, self.model_filename)
                    )
                    self.seg_module.model.load_state_dict(
                        state_dict=best_model_state_dict
                    )
                    LOGGER.info(
                        f"Test with .pth file :{os.path.join(self.output_folder, self.model_filename)}"
                    )

                if self.get_prediction:
                    self.trainer.predict(
                        model=self.seg_module,
                        datamodule=self.data_module,
                        ckpt_path=best_val_loss_ckpt_path,
                    )

                else:
                    self.trainer.test(
                        model=self.seg_module,
                        datamodule=self.data_module,
                        ckpt_path=best_val_loss_ckpt_path,
                    )

            except OdeonError as error:
                raise OdeonError(
                    ErrorCodes.ERR_TRAINING_ERROR,
                    "ERROR: Something went wrong during the test step of the training",
                    stack_trace=error,
                )

    def configure_loggers(self):
        loggers = []

        if self.use_tensorboard:
            train_logger = TensorBoardLogger(
                save_dir=os.path.join(self.output_folder, self.name_exp_log),
                name="tensorboard_logs",
                version=self.version_name,
                default_hp_metric=False,
                sub_dir="Train",
                filename_suffix="_train",
            )

            valid_logger = TensorBoardLogger(
                save_dir=os.path.join(self.output_folder, self.name_exp_log),
                name="tensorboard_logs",
                version=self.version_name,
                default_hp_metric=False,
                sub_dir="Validation",
                filename_suffix="_val",
            )
            loggers.extend([train_logger, valid_logger])

            if self.test_file:
                test_logger = TensorBoardLogger(
                    save_dir=os.path.join(self.output_folder, self.name_exp_log),
                    name="tensorboard_logs",
                    version=self.version_name,
                    default_hp_metric=False,
                    sub_dir="Test",
                    filename_suffix="_test",
                )
                loggers.append(test_logger)

        if self.save_history:
            json_logger = JSONLogger(
                save_dir=os.path.join(self.output_folder, self.name_exp_log),
                version=self.version_name,
                name="history_json",
            )
            loggers.append(json_logger)

            if self.test_file:
                test_json_logger = JSONLogger(
                    save_dir=os.path.join(self.output_folder, self.name_exp_log),
                    version=self.version_name,
                    name="test_json",
                )
                loggers.append(test_json_logger)

        if self.use_wandb:
            wandb_logger = WandbLogger(
                project=self.name_exp_log,
                save_dir=os.path.join(self.output_folder, self.name_exp_log),
            )
            loggers.append(wandb_logger)

        if self.verbosity:
            LOGGER.debug(f"DEBUG: Loggers: {loggers}")

        return loggers

    def configure_callbacks(self):
        # Callbacks definition
        callbacks = []
        if self.use_tensorboard:
            tensorboard_metrics = MetricsAdder()
            callbacks.append(tensorboard_metrics)

        if self.use_wandb:
            from odeon.callbacks.wandb import (
                LogConfusionMatrix,
                MetricsWandb,
                UploadCodeAsArtifact,
            )

            code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            callbacks.extend(
                [
                    MetricsWandb(),
                    LogConfusionMatrix(),
                    UploadCodeAsArtifact(code_dir=code_dir, use_git=True),
                ]
            )
        if self.model_out_ext == ".ckpt":
            checkpoint_miou_callback = LightningCheckpoint(
                monitor="val_miou",
                dirpath=os.path.join(
                    self.output_folder, self.name_exp_log, "odeon_miou_ckpt"
                ),
                version=self.version_name,
                filename=self.model_filename,
                save_top_k=self.save_top_k,
                mode="max",
                save_last=True,
            )

            checkpoint_loss_callback = LightningCheckpoint(
                monitor="val_loss",
                dirpath=os.path.join(
                    self.output_folder, self.name_exp_log, "odeon_val_loss_ckpt"
                ),
                version=self.version_name,
                filename=self.model_filename,
                save_top_k=self.save_top_k,
                mode="min",
                save_last=True,
            )
            callbacks.extend([checkpoint_miou_callback, checkpoint_loss_callback])

        elif self.model_out_ext == ".pth":
            checkpoint_pth = ExoticCheckPoint(
                out_folder=self.output_folder,
                out_filename=self.model_filename,
                model_out_ext=self.model_out_ext,
            )
            callbacks.append(checkpoint_pth)

        else:
            LOGGER.error(
                "ERROR: parameter model_out_ext could only be .ckpt or  .pth ..."
            )
            raise OdeonError(
                ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                "The input parameter model_out_ext is incorrect.",
            )
        if self.save_history:
            callbacks.append(HistorySaver())

        if self.use_tensorboard:
            callbacks.extend(
                [
                    GraphAdder(),
                    HistogramAdder(),
                    PredictionsAdder(),
                    HParamsAdder(),
                ]
            )

        if self.log_learning_rate:
            lr_monitor_callback = LearningRateMonitor(
                logging_interval="epoch", log_momentum=True
            )
            callbacks.append(lr_monitor_callback)

        if self.early_stopping:
            early_stop_callback = EarlyStopping(
                monitor=self.early_stopping["monitor"],
                min_delta=self.early_stopping["min_delta"],
                patience=self.early_stopping["patience"],
                mode=self.early_stopping["mode"],
                verbose=self.verbosity,
            )
            callbacks.append(early_stop_callback)

        if self.continue_training:
            file_exist(os.path.join(self.output_folder, self.model_filename))
            resume_file_ext = os.path.splitext(
                os.path.join(self.output_folder, self.model_filename)
            )[-1]
            if resume_file_ext == ".pth":
                continue_training_callback = ContinueTraining(
                    out_dir=self.output_folder,
                    out_filename=self.model_filename,
                    save_history=self.save_history,
                )
                callbacks.append(continue_training_callback)
            elif resume_file_ext == ".ckpt":
                self.resume_checkpoint = os.path.join(
                    self.output_folder, self.model_filename
                )
            else:
                LOGGER.error(
                    "ERROR: Odeon only handles files of type .pth or .ckpt \
                    for the continue training feature."
                )
                raise OdeonError(
                    ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                    "The parameter model_filename is incorrect in this case \
                    with the parameter continue_training as true.",
                )

        if self.get_prediction:
            path_predictions = os.path.join(
                self.output_folder, self.name_exp_log, "predictions", self.version_name
            )
            custom_pred_writer = PatchPredictionWriter(
                output_dir=path_predictions,
                output_type=self.prediction_output_type,
                write_interval="batch",
            )
            callbacks.append(custom_pred_writer)

        if self.progress_rate <= 0:
            self.enable_progress_bar = False
        else:
            progress_bar = TQDMProgressBar(refresh_rate=self.progress_rate)
            callbacks.append(progress_bar)
            self.enable_progress_bar = True

        if self.verbosity:
            LOGGER.debug(f"DEBUG: Callbacks: {callbacks}")
        return callbacks

    def setup(self):

        if self.model_out_ext is None:
            if self.model_filename is None:
                self.model_out_ext = MODEL_OUT_EXT
            else:
                self.model_out_ext = os.path.splitext(self.model_filename)[-1]
        else:
            self.model_out_ext = self.model_out_ext

        if self.name_exp_log is None:
            self.name_exp_log = (
                self.model_name + "_" + date.today().strftime("%b_%d_%Y")
            )
        else:
            self.name_exp_log = self.name_exp_log

        if self.version_name is None:
            self.version_name = self.get_version_name()
        else:
            self.version_name = self.version_name

        if self.random_seed is not None:
            seed_everything(self.random_seed, workers=True)

        if self.deterministic is True:
            torch.use_deterministic_algorithms(True)

        if self.use_wandb:
            try:
                os.system("wandb login")
                # os.system("wandb offline")  # To save wandb logs in offline mode (save code and metrics)
            except OdeonError as error:
                LOGGER.error(
                    "ERROR: WANDB function have been called but wandb package is not working"
                )
                raise OdeonError(
                    ErrorCodes.ERR_TRAINING_ERROR,
                    "something went wrong during training configuration",
                    stack_trace=error,
                )
        if self.verbosity:
            LOGGER.debug(f"DEBUG: output folder: {self.output_folder}")
            LOGGER.debug(f"DEBUG: model_out_ext: {self.model_out_ext}")
            LOGGER.debug(f"DEBUG: name_exp_log: {self.name_exp_log}")
            LOGGER.debug(f"DEBUG: version_name: {self.version_name}")

    def check(self):
        if self.model_name not in MODEL_LIST:
            raise OdeonError(
                message=f"the model name {self.model_name} does not exist",
                error_code=ErrorCodes.ERR_MODEL_ERROR,
            )
        try:
            dirs_exist([self.output_folder])
            if self.continue_training:
                file_exist(os.path.join(self.output_folder, self.model_filename))
        except OdeonError as error:
            raise OdeonError(
                ErrorCodes.ERR_TRAINING_ERROR,
                "something went wrong during training configuration",
                stack_trace=error,
            )

    def get_version_name(self):
        version_idx = None
        path = os.path.join(self.output_folder, self.name_exp_log)
        if not os.path.exists(path):
            version_idx = 0
        else:
            ckpt_path = os.path.join(path, "odeon_val_loss_ckpt")
            if "odeon_val_loss_ckpt" not in os.listdir(path):
                version_idx = 0
                os.makedirs(ckpt_path)
            else:
                list_ckpt_dir = [
                    x
                    for x in os.listdir(ckpt_path)
                    if os.path.isdir(os.path.join(ckpt_path, x))
                ]
                found_idx = [
                    int(name_dir.split("_")[-1])
                    for name_dir in list_ckpt_dir
                    if "version_" in name_dir
                ]
                version_idx = max(found_idx) + 1 if found_idx is not None else 0

        version_name = f"version_{str(version_idx)}"
        return version_name

    def get_path_best_ckpt(self, ckpt_folder, monitor="val_loss", mode="min"):
        def _get_value_monitor(input_str):
            return float(input_str.split(monitor)[-1][1:5])

        best_ckpt_path = None
        list_ckpt = os.listdir(ckpt_folder)
        if len(list_ckpt) == 1:
            best_ckpt_path = list_ckpt[0]
        else:
            list_ckpt = [x for x in list_ckpt if monitor in x]
            value_ckpt = np.array([_get_value_monitor(x) for x in list_ckpt])
            if mode == "min":
                best_ckpt_path = list_ckpt[np.argmin(value_ckpt)]
            else:
                best_ckpt_path = list_ckpt[np.argmax(value_ckpt)]
        return os.path.join(ckpt_folder, best_ckpt_path)
