from dataclasses import dataclass, field
from typing import Dict, Optional

from odeon.core.types import PARAMS

from ._modality import Modality, Sample


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class ModalityCollection:

    modality_params: Dict[str, PARAMS | Modality]
    geo_referenced: bool = False
    _modalities: Optional[Dict[str, Modality]] = field(init=False)

    def __post_init__(self):
        for k, v in self.modality_params.items():
            if isinstance(v, Modality):
                self._modalities[k] = v
            else:
                pass  # TODO add modality registry to instanciate modalities self._modalities[k] =

    def read(self,
             bounds=None,
             as_dict: bool = False,
             *args,
             **kwargs,) -> Sample | Optional[PARAMS]:
        ...

    def write(self, *args, **kwargs):
        ...
