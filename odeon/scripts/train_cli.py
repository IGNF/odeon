import os
from datetime import date
from time import gmtime, strftime
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
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
from odeon.commons.guard import dirs_exist
from odeon.callbacks.utils_callbacks import MyModelCheckpoint, HistorySaver
from odeon.callbacks.tensorboard_callbacks import (
    MetricsAdder,
    GraphAdder,
    HistogramAdder,
    PredictionsAdder,
    HParamsAdder
)
from odeon.nn.transforms import Compose, Rotation90, Radiometry, ToDoubleTensor
from odeon.nn.models import build_model, model_list

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
                 percentage_val=PERCENTAGE_VAL,
                 image_bands=None,
                 mask_bands=None,
                 class_labels=None,
                 model_filename=None,
                 load_pretrained=False,
                 epochs=NUM_EPOCHS,
                 batch_size=BATCH_SIZE,
                 patience=PATIENCE,
                 save_history=True,
                 continue_training=False,
                 loss="ce",
                 class_imbalance=None,
                 optimizer="adam",
                 data_augmentation=None,
                 device=None,
                 reproducible=True,
                 lr=LEARNING_RATE,
                 num_workers=NUM_WORKERS,
                 output_tensorboard_logs=None,
                 strategy=None,
                 val_check_interval=VAL_CHECK_INTERVAL,
                 name_exp_log=None,
                 log_histogram=False,
                 log_graph=False,
                 log_predictions=False,
                 log_learning_rate=False,
                 log_hparams=False,
                 use_wandb=False,
                 early_stopping=False,
                 ):
        self.train_file = train_file
        self.val_file = val_file
        self.percentage_val = percentage_val
        self.verbosity = verbosity
        self.model_name = model_name
        self.output_folder = output_folder        
        if output_tensorboard_logs is None:
            self.output_tensorboard_logs = output_folder
        else: 
            self.output_tensorboard_logs = output_tensorboard_logs
        self.reproducible = reproducible
        self.model_filename = model_filename if model_filename is not None else f"{model_name}.pth"
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_bands = image_bands
        self.mask_bands = mask_bands
        self.patience = patience
        self.save_history = save_history
        self.continue_training = continue_training
        self.loss_name = loss
        self.optimizer_name = optimizer
        self.learning_rate = lr
        self.class_imbalance = class_imbalance
        self.load_pretrained = load_pretrained
        self.num_workers = num_workers
        self.val_check_interval = val_check_interval
        self.log_histogram = log_histogram
        self.log_graph = log_graph
        self.log_predictions = log_predictions
        self.log_learning_rate = log_learning_rate
        self.log_hparams = log_hparams
        self.use_wandb = use_wandb
        self.early_stopping = early_stopping

        if name_exp_log is None:
            self.name_exp_log = self.model_name + "_" + date.today().strftime("%b_%d_%Y")
        else:
            self.name_exp_log = name_exp_log

        if reproducible is True:
            self.random_seed = RANDOM_SEED
            seed_everything(self.random_seed, workers=True)
            # Devrait être à True mais problème avec le calcul de cm dans torchmetrics
            self.deterministic = False
        else:
            self.random_seed = None
            self.deterministic = False

        if strategy == "ddp":
            strategy = DDPStrategy(find_unused_parameters=False)
        else:
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

        self.device = device

        self.data_module = SegDataModule(train_file=self.train_file,
                                         val_file=self.val_file,
                                         test_file=None,
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
                                         subset=True)

        self.callbacks = None

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
        except OdeonError as error:
            raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                             "something went wrong during training configuration",
                             stack_trace=error)

    def configure(self):
        self.model = build_model(model_name=self.model_name,
                                 n_channels=self.data_module.num_channels,
                                 n_classes=self.data_module.num_classes,
                                 continue_training=self.continue_training,
                                 load_pretrained=self.load_pretrained)

        self.seg_module = SegmentationTask(model=self.model,
                                           num_classes=self.data_module.num_classes,
                                           class_labels=self.class_labels,
                                           criterion=self.loss_name,
                                           optimizer=self.optimizer_name,
                                           learning_rate= self.learning_rate,
                                           patience=self.patience,
                                           weights=self.class_imbalance,
                                           log_histogram=self.log_histogram,
                                           log_graph=self.log_graph,
                                           log_predictions=self.log_predictions)
        # Loggers definition
        version_name = "version_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        train_logger = TensorBoardLogger(save_dir=os.path.join(self.output_tensorboard_logs, self.name_exp_log),
                                        name="tensorboard_logs",
                                        version=version_name,
                                        sub_dir='Train',
                                        filename_suffix='_train')

        valid_logger = TensorBoardLogger(save_dir=os.path.join(self.output_tensorboard_logs, self.name_exp_log),
                                        name="tensorboard_logs",
                                        version=version_name,
                                        sub_dir='Validation',
                                        filename_suffix='_val')

        loggers = [train_logger, valid_logger]

        if self.use_wandb:
            wandb_logger = WandbLogger(project=self.name_exp_log)
            loggers.append(wandb_logger)

        if self.save_history:
            csv_logger = CSVLogger(save_dir=os.path.join(self.output_folder, self.name_exp_log),
                                   version=version_name,
                                   name="logs")
            loggers.append(csv_logger)

        # Callbacks definition
        checkpoint_miou_callback = MyModelCheckpoint(monitor="val_miou",
                                                     dirpath=os.path.join(self.output_folder, self.name_exp_log, "odeon_miou_ckpt"),
                                                     version=version_name,
                                                     save_top_k=NUM_CKPT_SAVED,
                                                     mode="max")

        checkpoint_loss_callback = MyModelCheckpoint(monitor="val_loss",
                                                     dirpath=os.path.join(self.output_folder, self.name_exp_log, "odeon_val_loss_ckpt"),
                                                     version=version_name,
                                                     save_top_k=NUM_CKPT_SAVED,
                                                     mode="min")

        tensorboard_metrics = MetricsAdder()

        self.callbacks = [checkpoint_miou_callback, checkpoint_loss_callback, tensorboard_metrics]

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

        self.trainer = Trainer(val_check_interval=self.val_check_interval,
                               gpus=self.device,
                               callbacks=self.callbacks,
                               max_epochs=self.epochs,
                               logger=loggers,
                               deterministic=self.deterministic,
                               strategy=self.strategy)

    def __call__(self):
        """
            Call the Trainer
        """
        try:
            self.trainer.fit(self.seg_module, datamodule=self.data_module)
        except OdeonError as error:
            raise error
        except KeyboardInterrupt:
            tmp_file = os.path.join('/tmp', 'INTERRUPTED.pth')
            tmp_optimizer_file = os.path.join('/tmp', 'optimizer_INTERRUPTED.pth')
            torch.save(self.model.state_dict(), tmp_file)
            torch.save(self.optimizer_function.state_dict(), tmp_optimizer_file)
            STD_OUT_LOGGER.info(f"Saved interrupt as {tmp_file}")
