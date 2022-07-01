from dataclasses import dataclass, field
from typing import List
from .data import RASTER_ACCEPTED_EXTENSION


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class ManiFest:

    raster_accepted_prefix: List[str] = field(default_factory=lambda x: RASTER_ACCEPTED_EXTENSION)

    def __post_init__(self):

        self.raster_accepted_prefix = list(set(self.raster_accepted_prefix + RASTER_ACCEPTED_EXTENSION))
