from .core.data_flow import DataFlow


class Fetcher(DataFlow):

    def apply_raster(self, *args, **kwargs):
        ...
