from cd_ortho.core.data import InputDataKeys, InputDType

InputDataFields = {InputDataKeys.INPUT: {"name": "image", "dtype": InputDType.UINT8},
                   InputDataKeys.TARGET: {"name": "mask", "type": "mask", "encoding": "integer"},
                   InputDataKeys.PREDS: "preds",
                   InputDataKeys.METADATA: "metadata"}
