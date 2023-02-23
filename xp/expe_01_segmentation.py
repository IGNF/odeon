import os

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from odeon.data.data_module import Input
from odeon.models.change.module.change_unet import ChangeUnet

root: str = '/media/HP-2007S005-data'
root_dir: str = os.path.join(root, 'gers/change_dataset/patches')
fold_nb: int = 0
fold: str = f'fold-{fold_nb}'
root_fold: str = os.path.join(root_dir, fold)
dataset: str = os.path.join(root_fold, 'train_split_0.geojson')
fit_params = {'input_fields': {"T0": {"name": "T0", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3]},
                               "T1": {"name": "T1", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3]},
                               "mask": {"name": "change", "type": "mask", "encoding": "integer"}},
                               'input_file': dataset,
                               'root_dir': root_dir
              }
val_dataset: str = os.path.join(root_dir, 'val_split_0.geojson')
val_params = {'input_fields': {'T0': {"name": "T0", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3]},
                               'T1': {"name": "T1", "type": "raster", "dtype": "uint8", "band_indices": [1, 2, 3]},
                               'mask': {"name": "change", "type": "mask", "encoding": "integer"}},
                               'input_file': dataset,
                               'root_dir': root_dir
              }

input = Input(fit_params=fit_params,
              validate_params=val_params)
model = ChangeUnet(model='fc_siam_conc')
path_model_checkpoint = ''
save_top_k_models = 5
path_model_log = ''
accelerator = 'cpu'


def main():

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(dirpath=path_model_checkpoint,
                                       save_top_k=save_top_k_models,
                                       filename='epoch-{epoch}-loss-{val_iou:.2f}',
                                       mode="max",
                                       monitor='val_iou')
    callbacks = [lr_monitor, model_checkpoint]
    logger = pl_loggers.TensorBoardLogger(save_dir=path_model_log)
    trainer = Trainer(logger=logger, callbacks=callbacks, accelerator='cpu')
    trainer.fit(model=model, datamodule=input)


if __name__ == '__main__':

    main()
