from abc import ABCMeta


class BaseTransform(metaclass=ABCMeta):

    ...


class BasePreProcess(metaclass=ABCMeta):

    ...


class SampleWiseTransform(BaseTransform):

    def __init__(self, *args, **kwargs):
        ...

    def forward(self, *args, **kwargs):
        ...


class BatchWiseTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        ...

    def forward(self, *args, **kwargs):
        ...


class SampleWisePreProcess(BasePreProcess):
    def __init__(self, *args, **kwargs):
        ...

    def forward(self, *args, **kwargs):
        ...
