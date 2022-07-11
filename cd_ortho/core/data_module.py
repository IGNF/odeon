from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data import DataLoader, Dataset

from .dataframe import split_dataframe
from .input import InputDataFields
from .runner_utils import Stages
from .types import DATAFRAME, PREPROCESS_OPS, STAGES, URI_OR_URIS

from.dataframe import CSV_SUFFIX, create_pandas_dataframe_from_file, create_geopandas_dataframe_from_file


DEFAULT_DATALOADER_OPTIONS = {"batch_size": 8, "num_workers": 1}


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class Input(LightningDataModule):

    data_loaders: Dict[STAGES, DataLoader] = field(init=True)
    # transforms: Dict[STAGES, PREPROCESS_OPS] Could be useful in next future for batchwise transform
    # input_fields: Dict Could be useful

    @staticmethod
    def _init_preprocess(
            preprocess: Union[Dict[STAGES, PREPROCESS_OPS], PREPROCESS_OPS],
            stages: List[STAGES]) -> Dict[STAGES, PREPROCESS_OPS]:

        match preprocess:
            case dict():
                out: Dict[STAGES, PREPROCESS_OPS] = dict()
                for stage in stages:
                    if stage == STAGES.VALIDATE and stage not in preprocess.keys():
                        out[stage] = preprocess[Stages.FIT]
                    else:
                        out[stage] = preprocess[stage]
                    return out
            case _:
                return {stage: preprocess for stage in stages}

    @staticmethod
    def _build_dataframe_by_stage(input_data: Dict[STAGES, URI_OR_URIS],
                                  root_dir: Optional[URI_OR_URIS] = None,
                                  train_val_split: float = 0.8) -> Dict[STAGES, DATAFRAME]:
        """
        Parameters
        ----------
        input_data
        root_dir
        train_val_split

        Returns
        -------
         a dictionary of stage: DataFrame
        """

        if root_dir is not None:
            input_data = {stage: Path(root_dir) / uri for stage, uri in input_data.items()}
        data: Dict = dict()
        for stage, uri in input_data.items():

            if uri.endswith(CSV_SUFFIX):  # case Pandas DataFrame
                df = create_pandas_dataframe_from_file(uri)
                data[stage] = df
            else:  # case Geopandas DataFrame
                gdf = create_geopandas_dataframe_from_file(uri)
                data[stage] = gdf

        if STAGES.FIT in data.keys() and STAGES.VALIDATE not in data.keys():

            train, validation = split_dataframe(data[STAGES.FIT], split_ratio=train_val_split)
            data[STAGES.FIT] = train
            data[STAGES.VALIDATE] = validation
        return data

    @staticmethod
    def _build_dataset_by_stage(
            dataframes: Dict[STAGES, DATAFRAME],
            preprocess: PREPROCESS_OPS,
            input_fields: Dict,
            transforms: Union[Dict[STAGES, PREPROCESS_OPS], None]) -> Dict[STAGES, Dataset]:
        """

        Parameters
        ----------
        dataframes
        preprocess
        input_fields
        transforms

        Returns
        -------

        """
        # TODO
        ...

    @staticmethod
    def _build_dataloader_by_stage(
            datasets: Dict[STAGES, Dataset],
            dataloader_options: Union[Optional[Dict], Dict[STAGES, Dict]]) -> Dict[STAGES,
                                                                                   DataLoader]:
        data_loaders = dict()
        for stage, dataset in datasets.items():
            if stage not in dataloader_options.keys():
                options = DEFAULT_DATALOADER_OPTIONS
                options["shuffle"] = stage == STAGES.FIT
                data_loaders[stage] = DataLoader(dataset=dataset, **options)
            else:
                data_loaders[stage] = DataLoader(dataset, **dataloader_options)
        return data_loaders

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == Stages.FIT:
            pass

    def get_dataloader(self, stage: STAGES):
        return self.data_loaders[stage]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.get_dataloader(Stages.FIT)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(Stages.VALIDATE)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(Stages.TEST)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(Stages.PREDICT)

    @staticmethod
    def from_files(
            input_data: Dict[STAGES, URI_OR_URIS],
            preprocess: PREPROCESS_OPS,
            transforms: Union[Dict[STAGES, PREPROCESS_OPS], None] = None,
            input_fields: Optional[Dict] = None,
            root_dir: Optional[URI_OR_URIS] = None,
            dataloader_options: Union[Optional[Dict], Dict[STAGES, Dict]] = None,
            train_val_split: float = 0.8
    ) -> LightningDataModule:
        """
        Build the Input class which is responsable of
        input model data.
        Build Dataloader(s) for each asking Stage

        Parameters
        ----------
        input_data
        preprocess
        transforms
        input_fields
        root_dir
        dataloader_options
        train_val_split

        Returns
        -------

        """
        # TODO sanitize() function, check if input is

        input_fields = InputDataFields if input_fields is None else input_fields
        geo_data_frames = Input._build_dataframe_by_stage(input_data=input_data,
                                                          root_dir=root_dir,
                                                          train_val_split=train_val_split)
        datasets = Input._build_dataset_by_stage(dataframes=geo_data_frames,
                                                 preprocess=preprocess,
                                                 transforms=transforms,
                                                 input_fields=input_fields)

        dataloaders = Input._build_dataloader_by_stage(datasets=datasets,
                                                       dataloader_options=dataloader_options)
        return Input(data_loaders=dataloaders)
