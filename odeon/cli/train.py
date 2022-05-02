
import os
import albumentations as A
import numpy as np
import pandas as pd
from datetime import date
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import (
    Trainer,
    seed_everything
)
from odeon import LOGGER
from odeon.callbacks.legacy import ContinueTraining, ExoticCheckPoint
from odeon.callbacks.tensorboard import (
    MetricsAdder,
    GraphAdder,
    HistogramAdder,
    PredictionsAdder,
    HParamsAdder
)
from odeon.callbacks.history import HistorySaver
from odeon.callbacks.checkpoint import LightningCheckpoint
from odeon.callbacks.writer import PatchPredictionWriter
from odeon.commons.core import BaseTool
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.commons.guard import dirs_exist, file_exist
from odeon.data.datamodules.patch_datamodule import SegDataModule
from odeon.modules.seg_module import SegmentationTask
from odeon.metrics.stats_module import Stats
from odeon.data.transforms.base import (
    Compose, 
    Rotation90, 
    Radiometry,
    ToDoubleTensor,
    ScaleImageToFloat
)
from odeon.models.base import model_list
from odeon.loggers.json_logs import JSONLogger

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_training")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)
RANDOM_SEED = 42
PERCENTAGE_VAL = 0.3
VAL_CHECK_INTERVAL = 1.0
BATCH_SIZE = 5
PATIENCE = 30
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
NUM_WORKERS = 4
NUM_CKPT_SAVED = 3
MODEL_OUT_EXT = ".ckpt"
ACCELERATOR = "gpu"
PROGRESS = 1
NUM_NODES = 1


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
        verbosity,
        model_name,
        output_folder,
        train_file,
        val_file=None,
        test_file=None,
        percentage_val=PERCENTAGE_VAL,
        image_bands=None,
        mask_bands=None,
        class_labels=None,
        resolution=None,
        model_filename=None,
        model_out_ext=None,
        init_weights=None,
        compute_normalization_weights=False,
        normalization_weights=None,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        load_pretrained_weights=None,
        init_model_weights=None,
        save_history=True,
        continue_training=False,
        loss="ce",
        class_imbalance=None,
        optimizer_config=None,
        scheduler_config=None,
        data_augmentation=None,
        device=None,
        accelerator=ACCELERATOR,
        num_nodes=NUM_NODES,
        num_processes=None,
        reproducible=True,
        lr=LEARNING_RATE,
        num_workers=NUM_WORKERS,
        output_tensorboard_logs=None,
        strategy=None,
        val_check_interval=VAL_CHECK_INTERVAL,
        name_exp_log=None,
        version_name=None,
        log_histogram=False,
        log_graph=False,
        log_predictions=False,
        log_learning_rate=False,
        log_hparams=False,
        use_wandb=False,
        early_stopping=False,
        save_top_k=NUM_CKPT_SAVED,
        get_prediction=False,
        prediction_output_type="uint8",
        testing=False,
        progress=PROGRESS
        ):
    
        self.verbosity = verbosity

        # Parameters for outputs 
        self.output_folder = output_folder
        self.name_exp_log = name_exp_log
        self.output_tensorboard_logs = output_tensorboard_logs
        self.version_name = version_name
        self.model_filename = model_filename
        self.model_out_ext = model_out_ext

        # Computations of data stats (mean and std) in order to do normalization
        self.data_augmentation= data_augmentation
        self.compute_normalization_weights = compute_normalization_weights
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
        self.init_weights = init_weights
        self.init_model_weights = init_model_weights
        self.model_name = model_name
        self.load_pretrained_weights = load_pretrained_weights

        # Loggers
        self.log_histogram = log_histogram
        self.log_graph = log_graph
        self.log_predictions = log_predictions
        self.log_learning_rate = log_learning_rate
        self.log_hparams = log_hparams

        # Callbacks
        self.patience = patience
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
        self.reproducible= reproducible
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

        self.transforms = self.configure_transforms(self.data_augmentation)

        self.data_module = SegDataModule(train_file=self.train_file,
                                         val_file=self.val_file,
                                         test_file=self.test_file,
                                         image_bands=self.image_bands,
                                         mask_bands=self.mask_bands,
                                         transforms=self.transforms,
                                         width=None,
                                         height=None,
                                         batch_size=self.batch_size,
                                         num_workers=self.num_workers,
                                         percentage_val=self.percentage_val,
                                         pin_memory=True,
                                         deterministic=self.deterministic,
                                         get_sample_info=self.get_prediction,
                                         resolution=self.resolution,
                                         subset=self.testing)

        self.seg_module = SegmentationTask(model_name=self.model_name,
                                           num_classes=self.data_module.num_classes,
                                           num_channels=self.data_module.num_channels,
                                           class_labels=self.data_module.class_labels,
                                           criterion_name=self.loss_name,
                                           learning_rate= self.learning_rate,
                                           optimizer_config=self.optimizer_config,
                                           scheduler_config=self.scheduler_config,
                                           patience=self.patience,
                                           load_pretrained_weights=self.load_pretrained_weights,
                                           init_model_weights=self.init_model_weights,
                                           loss_classes_weights=self.class_imbalance)

        self.loggers = self.configure_loggers()

        self.callbacks = self.configure_callbacks()

        self.trainer = Trainer(val_check_interval=self.val_check_interval,
                               devices=self.device,
                               accelerator=self.accelerator,
                               callbacks=self.callbacks,
                               max_epochs=self.epochs,
                               logger=self.loggers,
                               deterministic=self.deterministic,
                               strategy=self.strategy,
                               num_nodes=self.num_nodes,
                               num_processes=self.num_processes,
                               enable_progress_bar=self.enable_progress_bar)

    def __call__(self):
        """
            Call the Trainer
        """
        self.configure()

        STD_OUT_LOGGER.info(
            f"Training : \n" 
            f"device: {self.device} \n"
            f"model: {self.model_name} \n"
            f"model file: {self.model_filename} \n"
            f"number of classes: {self.data_module.num_classes} \n"
            f"number of samples: {len(self.data_module.train_image_files) + len(self.data_module.val_image_files)}  "
            f"(train: {len(self.data_module.train_image_files)}, val: {len(self.data_module.val_image_files)})"
            )

        try:
            self.trainer.fit(self.seg_module,
                             datamodule=self.data_module,
                             ckpt_path=self.resume_checkpoint)

        except OdeonError as error:
            raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                             "ERROR: Something went wrong during the fit step of the training",
                             stack_trace=error)

        if self.test_file is not None:
            try:
                best_val_loss_ckpt_path = None
                if self.model_out_ext == ".ckpt":
                    ckpt_val_loss_folder = os.path.join(self.output_folder, self.name_exp_log, "odeon_val_loss_ckpt", self.version_name)
                    best_val_loss_ckpt_path = self.get_path_best_ckpt(ckpt_folder=ckpt_val_loss_folder,
                                                                      monitor="val_loss",
                                                                      mode="min")
 
                elif self.model_out_ext == ".pth":
                    # Load model weights into the model of the seg module
                    best_model_state_dict = torch.load(os.path.join(self.output_folder, self.model_filename))
                    self.seg_module.model.load_state_dict(state_dict=best_model_state_dict)
                    LOGGER.info(f"Test with .pth file :{os.path.join(self.output_folder, self.model_filename)}")

                if self.get_prediction:
                    self.trainer.predict(self.seg_module,
                                         datamodule=self.data_module,
                                         ckpt_path=best_val_loss_ckpt_path)

                else:
                    self.trainer.test(self.seg_module,
                                      datamodule=self.data_module,
                                      ckpt_path=best_val_loss_ckpt_path)

            except OdeonError as error:
                raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                                    "ERROR: Something went wrong during the test step of the training",
                                    stack_trace=error)

    def configure_transforms(self, data_aug):

        def _parse_data_augmentation(list_tfm):
            tfm_dict = {"rotation90": Rotation90(), "radiometry": Radiometry()}
            list_tfm = list_tfm if isinstance(list_tfm, list) else [list_tfm]
            tfm_func = [tfm_dict[tfm] for tfm in list_tfm]
            return tfm_func

        if self.compute_normalization_weights is True:
            self.normalization_weights = self.compute_stats()

        transforms = {}
        for split_name in ["train", "val", "test"]:
            tfm_func = [] if data_aug is None else _parse_data_augmentation(data_aug[split_name])
            # Part to define how to normalize the data
            if self.normalization_weights is not None:
                if isinstance(self.normalization_weights, dict):
                    self.normalization_weights = pd.DataFrame(self.normalization_weights).T
                if  split_name != "test" or (split_name == "test" and self.test_file is not None):
                    tfm_func.extend([A.Normalize(mean=self.normalization_weights.loc[split_name, "mean"],
                                                 std=self.normalization_weights.loc[split_name, "std"])])
            else:
                tfm_func.append(ScaleImageToFloat())
            tfm_func.append(ToDoubleTensor())  # To transform float type arrays to double type tensors
            transforms[split_name] = Compose(tfm_func)
        return transforms

    def configure_loggers(self):
        # Loggers definition
        train_logger = TensorBoardLogger(save_dir=os.path.join(self.output_tensorboard_logs, self.name_exp_log),
                                         name="tensorboard_logs",
                                         version=self.version_name,
                                         default_hp_metric=False,
                                         sub_dir='Train',
                                         filename_suffix='_train')

        valid_logger = TensorBoardLogger(save_dir=os.path.join(self.output_tensorboard_logs, self.name_exp_log),
                                         name="tensorboard_logs",
                                         version=self.version_name,
                                         default_hp_metric=False,
                                         sub_dir='Validation',
                                         filename_suffix='_val')

        loggers = [train_logger, valid_logger]

        if self.use_wandb:
            wandb_logger = WandbLogger(project=self.name_exp_log,
                                       save_dir=os.path.join(self.output_folder, self.name_exp_log))
            loggers.append(wandb_logger)

        if self.save_history:
            json_logger = JSONLogger(save_dir=os.path.join(self.output_folder, self.name_exp_log),
                                    version=self.version_name,
                                    name="history_json")
            loggers.append(json_logger)

        if self.test_file:
            # Logger will be use for test or predict phase
            test_logger = TensorBoardLogger(save_dir=os.path.join(self.output_tensorboard_logs, self.name_exp_log),
                                            name="tensorboard_logs",
                                            version=self.version_name,
                                            default_hp_metric=False,
                                            sub_dir='Test',
                                            filename_suffix='_test')
            loggers.append(test_logger)

            if self.save_history:
                test_json_logger = JSONLogger(save_dir=os.path.join(self.output_folder, self.name_exp_log),
                                              version=self.version_name,
                                              name="test_json")
                loggers.append(test_json_logger)
        return loggers

    def configure_callbacks(self):
        # Callbacks definition
        tensorboard_metrics = MetricsAdder()
        callbacks = [tensorboard_metrics]

        if self.use_wandb:
            from odeon.callbacks.wandb import (
                LogConfusionMatrix,
                MetricsWandb, 
                UploadCodeAsArtifact
            )
            code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            callbacks.extend([MetricsWandb(),
                                   LogConfusionMatrix(),
                                   UploadCodeAsArtifact(code_dir=code_dir,
                                                        use_git=True)])
        if self.model_out_ext == ".ckpt":
            checkpoint_miou_callback = LightningCheckpoint(monitor="val_miou",
                                                         dirpath=os.path.join(self.output_folder, self.name_exp_log, "odeon_miou_ckpt"),
                                                         version=self.version_name,
                                                         filename=self.model_filename,
                                                         save_top_k=self.save_top_k,
                                                         mode="max",
                                                         save_last=True)

            checkpoint_loss_callback = LightningCheckpoint(monitor="val_loss",
                                                         dirpath=os.path.join(self.output_folder, self.name_exp_log, "odeon_val_loss_ckpt"),
                                                         version=self.version_name,
                                                         filename=self.model_filename,
                                                         save_top_k=self.save_top_k,
                                                         mode="min",
                                                         save_last=True)
            callbacks.extend([checkpoint_miou_callback, checkpoint_loss_callback])
        elif self.model_out_ext == ".pth":
            checkpoint_pth = ExoticCheckPoint(out_folder=self.output_folder,
                                              out_filename=self.model_filename,
                                              model_out_ext=self.model_out_ext)
            callbacks.append(checkpoint_pth)
        else:
            LOGGER.error('ERROR: parameter model_out_ext could only be .ckpt or  .pth ...')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                "The input parameter model_out_ext is incorrect.")
        if self.save_history:
            callbacks.append(HistorySaver())
        if self.log_graph:
            callbacks.append(GraphAdder())
        if self.log_histogram:
            callbacks.append(HistogramAdder())
        if self.log_predictions:
            callbacks.append(PredictionsAdder())
        if self.log_learning_rate:
            lr_monitor_callback = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
            callbacks.append(lr_monitor_callback)
        if self.log_hparams:
            callbacks.append(HParamsAdder())
        if self.early_stopping:
            if isinstance(self.early_stopping, str):
                mode = 'min' if self.early_stopping.lower().endswith('loss') else 'max'
                early_stop_callback = EarlyStopping(monitor=self.early_stopping, min_delta=0.00, patience=self.patience, verbose=False, mode=mode)
            else:
                early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=self.patience, verbose=False, mode="min")
            callbacks.append(early_stop_callback)
        if self.continue_training:
            file_exist(os.path.join(self.output_folder, self.model_filename))
            resume_file_ext = os.path.splitext(os.path.join(self.output_folder, self.model_filename))[-1]
            if resume_file_ext == ".pth":
                continue_training_callback = ContinueTraining(out_dir=self.output_folder,
                                                              out_filename=self.model_filename,
                                                              save_history=self.save_history)
                callbacks.append(continue_training_callback)
            elif resume_file_ext == ".ckpt":
                self.resume_checkpoint = os.path.join(self.output_folder, self.model_filename)
            else:
                LOGGER.error("ERROR: Odeon only handles files of type .pth or .ckpt for the continue training feature.")
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR, 
                                 "The parameter model_filename is incorrect in this case with the parameter continue_training as true.")
        if self.get_prediction:
            path_predictions = os.path.join(self.output_folder, self.name_exp_log, "predictions", self.version_name)
            custom_pred_writer = PatchPredictionWriter(output_dir=path_predictions,
                                                        output_type=self.prediction_output_type,
                                                        write_interval="batch")
            callbacks.append(custom_pred_writer)
        if self.progress_rate <= 0:
            self.enable_progress_bar = False
        else:
            progress_bar = TQDMProgressBar(refresh_rate=self.progress_rate)
            callbacks.append(progress_bar)
            self.enable_progress_bar = True
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
            self.name_exp_log = self.model_name + "_" + date.today().strftime("%b_%d_%Y")
        else:
            self.name_exp_log = self.name_exp_log

        if self.output_tensorboard_logs is None:
            self.output_tensorboard_logs = self.output_folder
        else: 
            self.output_tensorboard_logs = self.output_tensorboard_logs

        if self.version_name is None:
            self.version_name = self.get_version_name()
        else:
            self.version_name = self.version_name

        if self.reproducible is True:
            self.random_seed = RANDOM_SEED
            seed_everything(self.random_seed, workers=True)
            self.deterministic = False  # Should be true but problem with confusion matrix calculation in torchmetrics
        else:
            self.random_seed = None
            self.deterministic = False

        if self.use_wandb:
            try:
                os.system("wandb login")
                # os.system("wandb offline")  # To save wandb logs in offline mode (save code and metrics)
            except OdeonError as error:
                LOGGER.error("ERROR: WANDB function have been called but wandb package is not working")
                raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                                 "something went wrong during training configuration",
                                 stack_trace=error)

    def check(self):
        if self.model_name not in model_list:
            raise OdeonError(message=f"the model name {self.model_name} does not exist",
                             error_code=ErrorCodes.ERR_MODEL_ERROR)
        try:
            dirs_exist([self.output_folder])
            if self.continue_training:
                file_exist(os.path.join(self.output_folder, self.model_filename))
        except OdeonError as error:
            raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                             "something went wrong during training configuration",
                             stack_trace=error)

    def compute_stats(self):
        stats = Stats(train_file=self.train_file,
                    val_file=self.val_file,
                    test_file=self.test_file,
                    image_bands=self.image_bands,
                    mask_bands=self.mask_bands,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    percentage_val=self.percentage_val,
                    deterministic=self.deterministic,
                    resolution=self.resolution,
                    subset=self.testing,
                    device=self.device,
                    accelerator=self.accelerator,
                    num_nodes=self.num_nodes,
                    num_processes=self.num_processes,
                    strategy=self.strategy)

        normalization_weights = stats()

        normalization_weights.to_csv(os.path.join(self.output_folder, "normalization_weights.csv"))
        return normalization_weights

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
                list_ckpt_dir = [x for x in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, x))]
                found_idx = [int(name_dir.split("_")[-1]) for name_dir in list_ckpt_dir if "version_" in name_dir]
                version_idx = max(found_idx) + 1 if found_idx is not None else 0

        version_name = f"version_{str(version_idx)}"
        return version_name

    def get_path_best_ckpt(self, ckpt_folder, monitor="val_loss", mode="min"):
        best_ckpt_path = None
        list_ckpt = os.listdir(ckpt_folder)
        if len(list_ckpt) == 1:
            best_ckpt_path = list_ckpt[0]
        else:
            list_ckpt = [x for x in list_ckpt if monitor in x]
            get_value_monitor = lambda x : float(x.split(monitor)[-1][1: 5])
            value_ckpt = np.array([get_value_monitor(x) for x in list_ckpt ])
            if mode == "min":
                best_ckpt_path = list_ckpt[np.argmin(value_ckpt)]
            else:
                best_ckpt_path = list_ckpt[np.argmax(value_ckpt)]
        return os.path.join(ckpt_folder, best_ckpt_path)