from enum import Enum, unique


@unique
class TaskType(Enum):
    SAMPLE = "sample"
    GENERATE = "generate"
    TRAIN = "train"
    TEST = "test"
    PRED = "pred"
    METRICS = "metrics"
    STATS = "stats"

    def __str__(self) -> str:
        return self.value