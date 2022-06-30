from abc import ABCMeta


class BaseTransform(metaclass=ABCMeta):

    ...


class BasePreProcess(metaclass=ABCMeta):

    ...


class PatchWiseTransform(BaseTransform):

    def __init__(self, *args, **kwargs):
        ...

    def forward(self, *args, **kwargs):
        ...


class BatchWiseTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        ...

    def forward(self, *args, **kwargs):
        ...


class PatchWisePreProcess(BasePreProcess):
    def __init__(self, *args, **kwargs):
        ...

    def forward(self, *args, **kwargs):
        ...
