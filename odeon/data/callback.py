from typing import Any, Dict

from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core.hooks import DataHooks
from torch.utils.data import Dataset


class DataCallback(Callback, DataHooks):  # type: ignore
    """Data Callback Api, built over the Callback API of pytorch Lightning,
    and DataHooks from `lightning.pytorch.core.hooks`.
    this class aims to build customizable data piepline by modifying
    dataset pipeline, transformation, preprocessing, loading to fit specific needs.
    It can be used to build an abstract data module, dataset and any class used
    during the data pipeline of a typical training process.
    """

    def on_setup_start(self, data_module: 'LightningDataModule', *args, **kwargs):
        """Called before any logic implemented """
        ...

    def on_setup_end(self, data_module: 'LightningDataModule', *args, **kwargs):
        """Called after all logic implemented """
        ...

    def on_init_dataset_start(self, dataset: 'Dataset', data_module: 'LightningDataModule',
                              stage: Any, *args, **kwargs):
        """Called before any logic implemented """
        ...

    def on_init_dataset_end(self, dataset: 'Dataset', data_module: 'LightningDataModule',
                            stage: Any, *args, **kwargs):
        """ Called after any logic implemented """
        ...

    def on_init_data_module_start(self, data_module: 'LightningDataModule',
                                  *args, **kwargs):
        """ Called before any logic implemented """
        ...

    def on_init_data_module_end(self, data_module: 'LightningDataModule', *args, **kwargs):
        """Called after any logic implemented """
        ...

    def on_before_transform(self, dataset: 'Dataset', data: Any, stage: Any, *args, **kwargs):
        """ called on transform pipeline start"""
        ...

    def on_after_transform(self, dataset: 'Dataset', data: Any, stage: Any, *args, **kwargs):
        """ called on transform pipeline end """
        ...

    def on_before_preprocessing(self, dataset: 'Dataset', data: Any, stage: Any, *args, **kwargs):
        """ called on preprocessing pipeline start """
        ...

    def on_after_preprocessing(self, dataset: 'Dataset', data: Any, stage: Any, *args, **kwargs):
        """ called on preprocessing pipeline end """
        ...

    def on_fetching_start(self, dataset: 'Dataset', meta: Any, *args, **kwargs):
        """Called before fetching data """
        ...

    def on_fetching_end(self, dataset: 'Dataset', meta: Any, *args, **kwargs):
        """Called after fetching data """
        ...

    def on_normalize_data_start(self, dataset: 'Dataset', data: Any, stage: Any, *args, **kwargs):
        """ Called before normalizing data """
        ...

    def on_normalize_data_end(self, dataset: 'Dataset', data: Any, stage: Any, *args, **kwargs):
        """ Called after normalizing data """
        ...

    def on_before_dataloader_init(self, data_module: 'LightningDataModule', dataloader_params: Dict[Any, Any],
                                  stage: Any, *args, **kwargs):
        """ Called before dataloader is initialized """
        ...

    def on_after_dataloader_init(self, data_module: 'LightningDataModule', dataloader_params: Dict[Any, Any],
                                 stage: Any, *args, **kwargs):
        """ Called after dataloader is initialized """
        ...

    def on_train_dataloader_start(self, data_module: 'LightningDataModule', *args, **kwargs):
        """ Called at get_val_dataloader start from lightning data module """
        ...

    def on_train_dataloader_end(self, data_module: 'LightningDataModule', *args, **kwargs):
        """ Called at get_val_dataloader end from lightning data module """
        ...

    def on_val_dataloader_start(self, data_module: 'LightningDataModule', *args, **kwargs):
        """ Called at get_val_dataloader start from lightning data module """
        ...

    def on_val_dataloader_end(self, data_module: 'LightningDataModule', *args, **kwargs):
        """ Called at get_val_dataloader end from lightning data module """
        ...

    def on_test_dataloader_start(self, data_module: 'LightningDataModule', *args, **kwargs):
        """ Called at get_test_dataloader start from lightning data module """
        ...

    def on_test_dataloader_end(self, data_module: 'LightningDataModule', *args, **kwargs):
        """ Called at get_test_dataloader end from lightning data module """
        ...

    def on_predict_dataloader_start(self, data_module: 'LightningDataModule', *args, **kwargs):
        """ Called at get_predict_dataloader start from lightning data module """
        ...

    def on_predict_dataloader_end(self, data_module: 'LightningDataModule', *args, **kwargs):
        """ Called at get_predict_dataloader end from lightning data module """
        ...
