import os
from datetime import date
from time import gmtime, strftime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.nn import functional as F
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import (
    Trainer,
    seed_everything
)

from odeon import LOGGER
from odeon.nn.unet import UNet
from odeon.modules.model_module import SegmentationTask
from odeon.modules.data_module import SegmentationDataModule


# Callbacks
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

# Define data module
path_data = "/home/dl/speillet/test_odeon_ocsng_32/dataset_ocsng_gers_naf_fold"
csv_train = os.path.join(path_data, 'train_4_fold_2-fold_3-fold_4.csv')
csv_val = os.path.join(path_data, 'val_4_fold_1.csv')
csv_test = os.path.join(path_data, 'test_4_fold_5.csv')

image_bands = [1,2,3]
mask_bands = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
classes_ids = {'batiment' : 0, 
               'zone_impermeable': 1,
               'zone_permeable': 2,
               'piscine': 3,
               'sol_nu': 4,
               'surface_eau': 5,
               'neige': 6,
               'coupe': 7,
               'peuplement_feuillus': 8,
               'peuplement_coniferes': 9,
               'lande_ligneuse': 10,
               'vigne': 11,
               'culture': 12,
               'terre_arable': 13,
               'autre': 14}
class_labels = classes_ids.keys()

batch_size = 2
num_workers = 4
pin_memory = True
testing = True

splits_csv = {'train': csv_train,
              'val': csv_val,
              'test': csv_test}

transforms = {'train': None,
              'val': None,
              'test': None}

ocsge_data_module = SegmentationDataModule(splits_csv=splits_csv,
                                           image_bands=image_bands,
                                           mask_bands=mask_bands,
                                           transforms=transforms,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           testing=testing)

# Model and hyperparameters definition
num_epochs = 10 + 1 # première epoch commence à 0
criterion = F.cross_entropy
optimizer = torch.optim.SGD
learning_rate = 0.02
num_classes = len(mask_bands)
unet = UNet(n_channels=len(image_bands),
            n_classes=num_classes)

val_check_interval = 0.5  # Number of validation in one epoch 0.5 => 2 val in 1 train

# Reproductibility - sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(42, workers=True)
deterministic=False # Devrait être à True mais problème avec le calcul de  cm dans torchmetrics

module = SegmentationTask(model=unet,
                          num_classes=num_classes,
                          class_labels=class_labels,
                          criterion=criterion,
                          optimizer=optimizer,
                          learning_rate=learning_rate,
                          val_check_interval=val_check_interval,
                          log_histogram=True,
                          log_graph=True)

model_name = "unet"
path_storage = os.getcwd()
path_logs = os.path.join(path_storage, "tensorboard_logs")
name_exp_log = model_name + "_" + date.today().strftime("%b_%d_%Y")

train_logger = TensorBoardLogger(save_dir=path_logs,
                                 name=name_exp_log,
                                 sub_dir='Train',
                                 filename_suffix='_train')

valid_logger = TensorBoardLogger(save_dir=path_logs,
                                 name=name_exp_log,
                                 sub_dir='Validation',
                                 filename_suffix='_val')

test_logger = TensorBoardLogger(save_dir=path_logs,
                                name=name_exp_log,
                                sub_dir='Test',
                                filename_suffix='_test')

loggers = [train_logger, valid_logger, test_logger]


strategy = DDPStrategy(find_unused_parameters=False)

# training
trainer = Trainer(val_check_interval=val_check_interval,
                  gpus=1,
                  callbacks=[checkpoint_miou_callback, checkpoint_loss_callback],
                  max_epochs=num_epochs,
                  logger=loggers,
                  deterministic=deterministic)

trainer.fit(module, datamodule=ocsge_data_module)