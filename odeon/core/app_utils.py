from enum import Enum


class Stages(Enum):
    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"


class Strategy(Enum):
    NORMAL = "normal"
    FINETUNE = "finetune"


L_STAGES = ['fit', 'validate', 'test', 'predict', Stages.FIT, Stages.VALIDATE, Stages.TEST, Stages.PREDICT]
