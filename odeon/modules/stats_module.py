import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.modules.datamodule import SegDataModule
from odeon.nn.transforms import ToDoubleTensor
from odeon.modules.metrics_module import MeanVector, IncrementalVariance

RANDOM_SEED = 42
BATCH_SIZE = 5
NUM_WORKERS = 4
PERCENTAGE_VAL = 0.3
ACCELERATOR = "gpu"
NUM_NODES = 1
STATS_STRATEGY = "ddp"


class StatsModule(pl.LightningModule):

    def __init__(self, num_channels, pixel_depth=255):
        super().__init__()
        self.num_channels = num_channels
        self.pixel_depth = pixel_depth
        self.means = torch.zeros(self.num_channels)
        self.stds = torch.zeros(self.num_channels)
        self.report = pd.DataFrame(columns=["mean", "std"],
                                   index=["train", "val", "test"], 
                                   dtype="object")

    def setup(self, stage):
        if stage == "fit":
            self.train_means = MeanVector(len_vector=self.num_channels)
            self.train_vars = IncrementalVariance(len_vector=self.num_channels)

            self.val_means = MeanVector(len_vector=self.num_channels)
            self.val_vars = IncrementalVariance(len_vector=self.num_channels)

        elif stage == "test":
            self.test_means = MeanVector(len_vector=self.num_channels)
            self.test_vars = IncrementalVariance(len_vector=self.num_channels)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        self.train_means.update(images.mean([0, 2, 3]))
        self.train_vars.update(images.transpose(0, 1).flatten(start_dim=1).transpose(0, 1))

    def training_epoch_end(self, outputs):
        self.report.loc["train", "mean"] = self.train_means.compute().cpu().numpy() / self.pixel_depth
        self.report.loc["train", "std"] = torch.sqrt(self.train_vars.compute()).cpu().numpy() / self.pixel_depth

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        self.val_means.update(images.mean([0, 2, 3]))
        self.val_vars.update(images.transpose(0, 1).flatten(start_dim=1).transpose(0, 1))

    def validation_epoch_end(self, outputs):
        self.report.loc["val", "mean"] = self.val_means.compute().cpu().numpy() / self.pixel_depth
        self.report.loc["val", "std"] = torch.sqrt(self.val_vars.compute()).cpu().numpy() / self.pixel_depth

    def test_step(self, batch, batch_idx):
        images = batch["image"]
        self.test_means.update(images.mean([0, 2, 3]))
        self.test_vars.update(images.transpose(0, 1).flatten(start_dim=1).transpose(0, 1))

    def test_epoch_end(self, outputs):
        self.report.loc["test", "mean"] = self.test_means.compute().cpu().numpy() / self.pixel_depth
        self.report.loc["test", "std"] = torch.sqrt(self.test_vars.compute()).cpu().numpy() / self.pixel_depth

    def configure_optimizers(self):
        return super().configure_optimizers()


class Stats:

    def __init__(self,
                 train_file=None,
                 val_file=None,
                 test_file=None,
                 image_bands=None,
                 mask_bands=None,
                 width=None,
                 height=None,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS,
                 percentage_val=PERCENTAGE_VAL,
                 pin_memory=True,
                 deterministic=False,
                 resolution=None,
                 subset=False,
                 device=None,
                 accelerator=ACCELERATOR,
                 num_nodes=NUM_NODES,
                 num_processes=None,
                 strategy=None):

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.image_bands = image_bands
        self.mask_bands = mask_bands
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.percentage_val = percentage_val
        self.pin_memory = pin_memory
        self.deterministic = deterministic
        self.resolution = resolution
        self.subset = subset
        self.device = device
        self.accelerator = accelerator
        self.num_nodes = num_nodes
        self.num_processes = num_processes
        self.strategy = STATS_STRATEGY if strategy is not None else strategy

        self.transforms = {phase: ToDoubleTensor() for phase in ["train", "val", "test"]}

        self.data_module = SegDataModule(train_file=self.train_file,
                                         val_file=self.val_file,
                                         test_file=self.test_file,
                                         image_bands=self.image_bands,
                                         mask_bands=self.mask_bands,
                                         transforms=self.transforms,
                                         width=self.width,
                                         height=self.height,
                                         batch_size=self.batch_size,
                                         num_workers=self.num_workers,
                                         pin_memory=True,
                                         deterministic=self.deterministic,
                                         resolution=self.resolution,
                                         subset=self.subset,
                                         drop_last=True)

        self.devices = device
        self.accelerator = accelerator
        self.deterministic = deterministic
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.num_processes = num_processes

        self.configure()

    def configure(self):

        self.stats_module = StatsModule(num_channels=self.data_module.num_channels)

        self.trainer = Trainer(devices=self.devices,
                               accelerator=self.accelerator,
                               deterministic=self.deterministic,
                               strategy=self.strategy,
                               max_epochs=1,
                               num_nodes=self.num_nodes,
                               num_processes=self.num_processes)

    def __call__(self):
        try:
            self.trainer.fit(self.stats_module,
                             datamodule=self.data_module)
    
            if self.data_module.test_file is not None:
                self.trainer.test(self.stats_module,
                                  datamodule=self.data_module)
            return self.stats_module.report

        except OdeonError as error:
            raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                                "ERROR: Something went wrong during the test step of the training",
                                stack_trace=error)
