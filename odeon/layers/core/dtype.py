from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Union

import numpy as np

F16I = np.finfo("float16")
F32I = np.finfo("float32")
F64I = np.finfo("float64")


class DtypeProtocol(Protocol):

    def get_max_value(self) -> Union[int, float]:
        raise NotImplementedError("When you implement the DtypeProtocol, you must override this method.")

    def get_name(self):
        raise NotImplementedError("When you implement the DtypeProtocol, you must override this method.")

    def is_float(self) -> bool:
        raise NotImplementedError("When you implement the DtypeProtocol, you must override this method.")

    def __str__(self):
        raise NotImplementedError("When you implement the DtypeProtocol, you must override this method.")


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True)
class DType(ABC, DtypeProtocol):
    name: str = field(init=True)
    max_value: Union[int, float] = field(init=True)
    min_value: Union[int, float] = field(init=True)
    type_is_float: bool = field(init=True)

    def __str__(self):
        return self.name

    def get_max_value(self) -> Union[int, float]:
        return self.max_value

    def get_min_value(self) -> Union[int, float]:
        return self.max_value

    def get_name(self):
        return self.name

    def is_float(self) -> bool:
        return self.type_is_float


DTYPE_LIST: List[DType] = [DType(name='uint8', min_value=0, max_value=255, type_is_float=False),
                           DType(name='uint16', min_value=0, max_value=65535, type_is_float=False),
                           DType(name='uint32', min_value=0, max_value=4294967295, type_is_float=False),
                           DType(name='int16', min_value=-32768, max_value=32767, type_is_float=False),
                           DType(name='int32', min_value=-2147483648, max_value=2147483647, type_is_float=False),
                           DType(name='float16', min_value=float(F16I.min), max_value=float(F16I.max),
                                 type_is_float=True),
                           DType(name='float32', min_value=float(F32I.min), max_value=float(F32I.max),
                                 type_is_float=True),
                           DType(name='float64', min_value=float(F64I.max), max_value=float(F64I.max),
                                 type_is_float=True),]


DTYPE_DICT: Dict = {dt.name: dt for dt in DTYPE_LIST}


def dtype_factory(name: str | None,
                  dt: DType | None = None,
                  min_value: Union[int, float] = 0,
                  max_value: Union[int, float] = 255,
                  type_is_float: bool = False) -> DType:
    """
    Factory method for creating a Dtype instance
    Parameters
    ----------
    name: str | None
    dt: DType | None
    min_value: int | float, default 0
    max_value: int | float, default 255
    type_is_float: bool, default False

    Returns
    -------
    Dtype
    """
    if dt is None:
        if name is not None and name in DTYPE_DICT.keys():
            return DTYPE_DICT[name]
        else:
            return DType(name='uint8', min_value=min_value, max_value=max_value, type_is_float=type_is_float)
    else:
        return dt
