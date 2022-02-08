import os
from datetime import date
from random import sample
from time import gmtime, strftime
from numpy import integer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.nn import functional as F
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from odeon.scripts.stats import NUM_WORKERS
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import (
    Trainer,
    seed_everything
)
from odeon import LOGGER
from odeon.modules.seg_module import SegmentationTask
from odeon.modules.data_module import SegDataModule
from odeon.commons.core import BaseTool
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.commons.guard import dirs_exist
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


class CLITrain(BaseTool):
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
                 load_pretrained_enc=False,
                 epochs=NUM_EPOCHS,
                 batch_size=BATCH_SIZE,
                 patience=PATIENCE,
                 save_history=False,
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
                 ignore_index=None,
                 val_check_interval=VAL_CHECK_INTERVAL,
                 log_histogram=False,
                 log_graph=False,
                 log_predictions=False
                 ):
        self.train_file = train_file
        self.val_file = val_file
        self.percentage_val = percentage_val
        self.verbosity = verbosity
        self.model_name = model_name
        self.output_folder = output_folder        
        if output_tensorboard_logs is None:
            self.output_tensorboard_logs = os.path.join(output_folder, "tensorboard_logs")
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
        self.load_pretrained_enc = load_pretrained_enc
        self.num_workers = num_workers
        self.ignore_index = ignore_index
        self.val_check_interval = val_check_interval
        self.log_histogram = log_histogram
        self.log_graph = log_graph
        self.log_predictions = log_predictions

        if reproducible is True:
            self.random_seed = RANDOM_SEED
            seed_everything(self.random_seed, workers=True)
            # Devrait être à True mais problème avec le calcul de  cm dans torchmetrics
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
        self.model = build_model(self.model_name,
                                 self.data_module.num_channels,
                                 self.data_module.num_classes)

        self.seg_module = SegmentationTask(model=self.model,
                                           num_classes=self.data_module.num_classes,
                                           class_labels=self.class_labels,
                                           criterion=self.loss_name,
                                           optimizer=self.optimizer_name,
                                           learning_rate= self.learning_rate,
                                           patience=self.patience,
                                           weights=self.class_imbalance,
                                           ignore_index=self.ignore_index,
                                           val_check_interval=self.val_check_interval,
                                           log_histogram=self.log_histogram,
                                           log_graph=self.log_graph,
                                           log_predictions=self.log_predictions)

        def check_path_ckpt(path, description=None): 
            path_ckpt = None
            if not os.path.exists(path):
                path_ckpt = path
            else:
                description = description if description is not None else ""
                path_ckpt = os.path.join(path, description + "_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
                os.makedirs(path_ckpt)
            return path_ckpt

        ckpt_descript = f"test_pl"
        checkpoint_miou_callback = ModelCheckpoint(monitor="val_miou",
                                                dirpath=check_path_ckpt("odeon_miou_ckpt", description=ckpt_descript),
                                                filename="sample-test-{epoch:02d}-{val_miou:.2f}",
                                                save_top_k=3,
                                                mode="max")

        checkpoint_loss_callback = ModelCheckpoint(monitor="val_loss",
                                                dirpath=check_path_ckpt("odeon_loss_ckpt", description=ckpt_descript),
                                                filename="sample-test-{epoch:02d}-{val_loss:.2f}",
                                                save_top_k=3,
                                                mode="min")

        name_exp_log = self.model_name + "_" + date.today().strftime("%b_%d_%Y")
        train_logger = TensorBoardLogger(save_dir=self.output_tensorboard_logs,
                                        name=name_exp_log,
                                        sub_dir='Train',
                                        filename_suffix='_train')

        valid_logger = TensorBoardLogger(save_dir=self.output_tensorboard_logs,
                                        name=name_exp_log,
                                        sub_dir='Validation',
                                        filename_suffix='_val')

        loggers = [train_logger, valid_logger]

        self.trainer = Trainer(val_check_interval=self.val_check_interval,
                               gpus=1,
                               callbacks=[checkpoint_miou_callback, checkpoint_loss_callback],
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
