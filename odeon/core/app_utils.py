from enum import Enum


class Stages(Enum):
    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"


class Strategy(Enum):
    NORMAL = "normal"
    FINETUNE = "finetune"
