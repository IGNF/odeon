from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from typing import Dict, Union, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import geopandas as gpd
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .runner_utils import Stages
from .types import URI_OR_URIS, STAGES, PREPROCESS_OPS, DATAFRAME
from .input import InputDataFields


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class Input(LightningDataModule):

    dataloaders: Dict[STAGES, DataLoader]
    transforms: Dict[STAGES, PREPROCESS_OPS]
    input_fields: Dict

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
    def _init_input():
        # TODO
        ...

    @staticmethod
    def _init_transform(
                        transform: Union[Dict[STAGES, PREPROCESS_OPS], PREPROCESS_OPS],
                        stages: List[STAGES]) -> Dict[STAGES, PREPROCESS_OPS]:

        return Input._init_preprocess(preprocess=transform, stages=stages)

    @staticmethod
    def _init_dataloaders(params: Union[Optional[Dict], Dict[STAGES, Dict]],
                          stages: List[STAGES]
                          ) -> Union[Optional[Dict], Dict[STAGES, Dict]]:

        match params:
            case dict():
                out: Union[Optional[Dict], Dict[STAGES, Dict]] = dict()
                for stage in stages:
                    if stage == STAGES.VALIDATE and stage not in params.keys():
                        out[stage] = params[Stages.FIT]
                    else:
                        out[stage] = params[stage]
                    return out
            case _:
                return {stage: params for stage in stages}

    @staticmethod
    def build_dataframe_by_stage(input_data,
                                 root_dir,
                                 input_fields,
                                 train_val_split=0.8) -> Dict[DATAFRAME]:
        """

        Parameters
        ----------
        input_data
        root_dir
        input_fields
        train_val_split

        Returns
        -------
         a dictionary of stage: DataFrame
        """
        # TODO
        ...

    @staticmethod
    def build_dataset_by_stage(self, dataframes: Dict[STAGES, DATAFRAME], ):
        # TODO
        ...

    @staticmethod
    def build_dataloader_by_stage(self):
        # TODO
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == STAGES.FIT:
            pass

    def get_dataloader(self, stage: STAGES):
        return self.dataloaders[stage]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.get_dataloader(STAGES.FIT)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(STAGES.VALIDATE)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(STAGES.TEST)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(STAGES.PREDICT)

    @staticmethod
    def from_vector_files(
            input_data: Dict[STAGES, URI_OR_URIS],
            preprocess: PREPROCESS_OPS,
            transform: Union[Dict[STAGES, PREPROCESS_OPS], Union[PREPROCESS_OPS]] = None,
            input_fields: Dict[STAGES, Any] = None,
            root_dir: Optional[URI_OR_URIS] = None,
            dataloader_options: Union[Optional[Dict], Dict[STAGES, Dict]] = None,
            train_val_split: float = 0.8
    ):
        """
        TODO sanitize() function, check if input is
        TODO build_geo_dataframe_by_stage,
         input: input_data, root_dir, train_val_split, input_fields => Dict[gpd.GeoDataFrame] as geo_dataframes
        TODO build_dataset_by_stage,
         input: dataframes, preprocess, transform => Dict[STAGES, Datasets] as datasets
        TODO build_dataloader_by_stage func,
         input: datasets, dataloader_options
        Parameters
        ----------
        input_data
        preprocess
        transform
        input_fields
        root_dir
        dataloader_options
        train_val_split

        Returns
        -------

        """

        input_fields = InputDataFields if input_fields is None else input_fields
        geo_data_frames = Input.build_dataframe_by_stage(input_data=input_data,
                                                         root_dir=root_dir,
                                                         input_fields=input_fields,
                                                         train_val_split=train_val_split)
        datasets = Input.build_dataset_by_stage()