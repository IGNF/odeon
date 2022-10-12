from odeon.core.data import InputDataKeys, InputDType, TargetTYPES

InputDataFields = {InputDataKeys.INPUT: {"name": "image", "type": "raster", "dtype": InputDType.UINT8},
                   InputDataKeys.TARGET: {"name": "mask", "type": TargetTYPES.MASK, "encoding": "integer"}
                   }
