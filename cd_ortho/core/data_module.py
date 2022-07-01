from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from typing import Dict, Union, List, Optional, Any, Sequence
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .runner_utils import Stages
from .types import URI_OR_URIS, SAMPLEWISE_OPS, STAGES
from .input import InputDataFields


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class Input(LightningDataModule):

    input_data: Dict[STAGES, URI_OR_URIS]
    preprocess: Union[Dict[STAGES, SAMPLEWISE_OPS], SAMPLEWISE_OPS]
    stages: Union[STAGES, List[STAGES]] = Stages.FIT
    input_fields: Dict[STAGES, Any] = None
    root_dir: Optional[URI_OR_URIS] = None
    transform: Union[Dict[STAGES, SAMPLEWISE_OPS], SAMPLEWISE_OPS] = None
    transform_strategy = "sample_wise"
    dataloader_options: Union[Optional[Dict], Dict[STAGES, Dict]] = None
    train_val_split: float = 0.8
    _data_loaders: Dict[STAGES, Union[DataLoader, Sequence[DataLoader]]] = field(init=False)

    def __post__init(self):

        self.stages = self.stages if isinstance(self.stages, List) else list(self.stages)
        self.input_fields = InputDataFields if self.input_fields is not None else self.input_fields

    @staticmethod
    def _init_preprocess(
                         preprocess: Union[Dict[STAGES, SAMPLEWISE_OPS], SAMPLEWISE_OPS],
                         stages: List[STAGES]) -> Dict[STAGES, SAMPLEWISE_OPS]:

        match preprocess:
            case dict():
                out: Dict[STAGES, SAMPLEWISE_OPS] = dict()
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
                        transform: Union[Dict[STAGES, SAMPLEWISE_OPS], SAMPLEWISE_OPS],
                        stages: List[STAGES]) -> Dict[STAGES, SAMPLEWISE_OPS]:

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
    def build_dataset(input: Dict[STAGES, URI_OR_URIS]):
        # TODO
        ...

    @staticmethod
    def build_dataloader(self):
        # TODO
        ...

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == STAGES.FIT:
            pass

    def get_dataloader(self, stage: STAGES):

        return self._data_loaders[stage]

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        return self.get_dataloader(STAGES.FIT)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(STAGES.VALIDATE)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(STAGES.TEST)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(STAGES.PREDICT)
