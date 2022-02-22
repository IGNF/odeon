import os
from datetime import date
from time import gmtime, strftime
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning import (
    Trainer,
    seed_everything
)
from odeon import LOGGER
from odeon.modules.seg_module import SegmentationTask
from odeon.modules.datamodule import SegDataModule
from odeon.commons.core import BaseTool
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.commons.guard import dirs_exist, files_exist, file_exist
from odeon.nn.models import get_train_filenames
from odeon.callbacks.utils_callbacks import ContinueTraining, HistorySaver, MyModelCheckpoint, ExoticCheckPoint
from odeon.callbacks.tensorboard_callbacks import (
    MetricsAdder,
    GraphAdder,
    HistogramAdder,
    PredictionsAdder,
    HParamsAdder
)
from odeon.nn.transforms import Compose, Rotation90, Radiometry, ToDoubleTensor
from odeon.nn.models import model_list

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
UNIQUE_CKPT = 1
MODEL_OUT_EXT = ".ckpt"
ACCELERATOR = "gpu"


class TrainCLI(BaseTool):
    """
    Main entry point of training tool

    Implements
    ----------
    BaseTool : object
        the abstract class for implementing a CLI tool
    """
    def __init__(self,
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
                 model_filename=None,
                 model_out_ext=None,
                 init_weights=None,
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
                 num_nodes=1,
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
                 testing=False
                 ):

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.percentage_val = percentage_val
        self.verbosity = verbosity
        self.model_filename = model_filename

        if model_out_ext is None:
            if self.model_filename is None:
                self.model_out_ext = MODEL_OUT_EXT
            else:
                self.model_out_ext = os.path.splitext(self.model_filename)[-1]
        else:
            self.model_out_ext = model_out_ext

        self.model_out_ext = model_out_ext
        self.model_name = model_name
        self.reproducible = reproducible
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_bands = image_bands
        self.mask_bands = mask_bands
        self.patience = patience
        self.load_pretrained_weights = load_pretrained_weights
        self.init_model_weights = init_model_weights
        self.save_history = save_history
        self.continue_training = continue_training
        self.loss_name = loss
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.learning_rate = lr
        self.class_imbalance = class_imbalance
        self.init_weights = init_weights
        self.num_workers = num_workers
        self.val_check_interval = val_check_interval
        self.log_histogram = log_histogram
        self.log_graph = log_graph
        self.log_predictions = log_predictions
        self.log_learning_rate = log_learning_rate
        self.log_hparams = log_hparams
        self.use_wandb = use_wandb
        self.early_stopping = early_stopping
        self.save_top_k = save_top_k
        self.testing = testing

        if name_exp_log is None:
            self.name_exp_log = self.model_name + "_" + date.today().strftime("%b_%d_%Y")
        else:
            self.name_exp_log = name_exp_log

        self.output_folder = output_folder   
        if output_tensorboard_logs is None:
            self.output_tensorboard_logs = output_folder
        else: 
            self.output_tensorboard_logs = output_tensorboard_logs

        if version_name is None:
            self.version_name = "_".join(["version",
                                          self.model_name,
                                          f"LR{str(self.learning_rate)}",
                                          strftime("%Y-%m-%d_%H-%M-%S", gmtime())])
        else:
            self.version_name = version_name

        if reproducible is True:
            self.random_seed = RANDOM_SEED
            seed_everything(self.random_seed, workers=True)
            # Devrait être à True mais problème avec le calcul de cm dans torchmetrics
            self.deterministic = False
        else:
            self.random_seed = None
            self.deterministic = False

        if self.use_wandb:
            os.system("wandb login")

        # if strategy == "ddp":
        #     strategy = DDPStrategy(find_unused_parameters=False)
        # else:
        #     self.strategy = strategy
        self.strategy = strategy

        if data_augmentation is None:
            self.transformation_functions = [ToDoubleTensor()]
        else:
            transformation_dict = {"rotation90": Rotation90(),
                                   "radiometry": Radiometry()}
            transformation_conf = data_augmentation
            transformation_keys = transformation_conf if isinstance(transformation_conf, list) else [transformation_conf]
            self.transformation_functions = list({
                value for key, value in transformation_dict.items() if key in transformation_keys
            })
            self.transformation_functions.append(ToDoubleTensor())

        self.transforms = {'train': Compose(self.transformation_functions),
                           'val': Compose(self.transformation_functions),
                           'test':Compose(self.transformation_functions)}

        self.accelerator = accelerator
        self.num_nodes = num_nodes
        self.num_processes = num_processes
        self.device = device

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
                                         subset=self.testing)

        self.callbacks = None
        self.resume_checkpoint = None
        self.train_files = None

        if class_labels is not None:
            if len(class_labels) == self.data_module.num_classes:
                self.class_labels = class_labels
            else:
                LOGGER.error('ERROR: parameter labels should have a number of values equal to the number of classes.')
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                    "The input parameter labels is incorrect.")
        else:
            self.class_labels = [f'class {i + 1}' for i in range(self.data_module.num_classes)]

        STD_OUT_LOGGER.info(f"""Training :
device: {self.device}
model: {self.model_name}
model file: {self.model_filename}
number of classes: {self.data_module.num_classes}
number of samples: {len(self.data_module.train_image_files) + len(self.data_module.val_image_files)} \
(train: {len(self.data_module.train_image_files)}, val: {len(self.data_module.val_image_files)})
""")
        self.check()
        self.configure()

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

    def configure(self):

        self.seg_module = SegmentationTask(model_name=self.model_name,
                                           num_classes=self.data_module.num_classes,
                                           num_channels=self.data_module.num_channels,
                                           class_labels=self.class_labels,
                                           criterion_name=self.loss_name,
                                           learning_rate= self.learning_rate,
                                           optimizer_config=self.optimizer_config,
                                           scheduler_config=self.scheduler_config,
                                           patience=self.patience,
                                           load_pretrained_weights=self.load_pretrained_weights,
                                           init_model_weights=self.init_model_weights,
                                           loss_classes_weights=self.class_imbalance)

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
            wandb_logger = WandbLogger(project=self.name_exp_log)
            loggers.append(wandb_logger)

        if self.save_history:
            csv_logger = CSVLogger(save_dir=os.path.join(self.output_folder, self.name_exp_log),
                                   version=self.version_name,
                                   name="history_csv")
            loggers.append(csv_logger)

        if self.test_file:
            test_logger = TensorBoardLogger(save_dir=os.path.join(self.output_tensorboard_logs, self.name_exp_log),
                                            name="tensorboard_logs",
                                            version=self.version_name,
                                            default_hp_metric=False,
                                            sub_dir='Test',
                                            filename_suffix='_test')
            loggers.append(test_logger)

            if self.save_history:
                test_csv_logger = CSVLogger(save_dir=os.path.join(self.output_folder, self.name_exp_log),
                                            version=self.version_name,
                                            name="test_csv")
                loggers.append(test_csv_logger)

        # Callbacks definition
        tensorboard_metrics = MetricsAdder()
        self.callbacks = [tensorboard_metrics]

        if self.model_out_ext == ".ckpt":
            checkpoint_miou_callback = MyModelCheckpoint(monitor="val_miou",
                                                         dirpath=os.path.join(self.output_folder, self.name_exp_log, "odeon_miou_ckpt"),
                                                         version=self.version_name,
                                                         filename=self.model_filename,
                                                         save_top_k=self.save_top_k,
                                                         mode="max",
                                                         save_last=True)

            checkpoint_loss_callback = MyModelCheckpoint(monitor="val_loss",
                                                         dirpath=os.path.join(self.output_folder, self.name_exp_log, "odeon_val_loss_ckpt"),
                                                         version=self.version_name,
                                                         filename=self.model_filename,
                                                         save_top_k=self.save_top_k,
                                                         mode="min",
                                                         save_last=True)
            self.callbacks.extend([checkpoint_miou_callback, checkpoint_loss_callback])
  
        elif self.model_out_ext == ".pth":
            checkpoint_pth = ExoticCheckPoint(out_folder=self.output_folder,
                                              out_filename=self.model_filename,
                                              model_out_ext=self.model_out_ext)
            self.callbacks.append(checkpoint_pth)

        else:
            LOGGER.error('ERROR: parameter model_out_ext could only be .ckpt or  .pth ...')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                "The input parameter model_out_ext is incorrect.")

        if self.save_history:
            self.callbacks.append(HistorySaver())

        if self.log_graph:
            self.callbacks.append(GraphAdder())

        if self.log_histogram:
            self.callbacks.append(HistogramAdder())

        if self.log_predictions:
            self.callbacks.append(PredictionsAdder())

        if self.log_learning_rate:
            lr_monitor_callback = LearningRateMonitor(logging_interval="step", log_momentum=True)
            self.callbacks.append(lr_monitor_callback)

        if self.log_hparams:
            self.callbacks.append(HParamsAdder())            

        if self.early_stopping:
            if isinstance(self.early_stopping, str):
                mode = 'min' if self.early_stopping.lower().endswith('loss') else 'max'
                early_stop_callback = EarlyStopping(monitor=self.early_stopping, min_delta=0.00, patience=self.patience, verbose=False, mode=mode)
            else:
                early_stop_callback = EarlyStopping(monitor="val_miou", min_delta=0.00, patience=self.patience, verbose=False, mode="max")
            self.callbacks.append(early_stop_callback)

        if self.continue_training:
            file_exist(os.path.join(self.output_folder, self.model_filename))
            ext = os.path.splitext(os.path.join(self.output_folder, self.model_filename))[-1]
            if ext == ".pth":
                continue_training_callback = ContinueTraining(out_dir=self.output_folder,
                                                              out_filename=self.model_filename,
                                                              save_history=self.save_history)
                self.callbacks.append(continue_training_callback)
            elif ext == ".ckpt":
                self.resume_checkpoint = os.path.join(self.output_folder, self.model_filename)
            else:
                LOGGER.error("ERROR: Odeon only handles files of type .pth, .ckpt for the continue training feature.")
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR, 
                                 "The parameter model_filename is incorrect in this case with the parameter continue_training as true.")

        self.trainer = Trainer(val_check_interval=self.val_check_interval,
                               devices=self.device,
                               accelerator=self.accelerator,
                               callbacks=self.callbacks,
                               max_epochs=self.epochs,
                               logger=loggers,
                               deterministic=self.deterministic,
                               strategy=self.strategy,
                               num_nodes=self.num_nodes,
                               num_processes=self.num_processes)

    def __call__(self):
        """
            Call the Trainer
        """
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
                self.trainer.test(self.seg_module,
                                  datamodule=self.data_module)
            except OdeonError as error:
                raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                                "ERROR: Something went wrong during the test step of the training",
                                stack_trace=error)
