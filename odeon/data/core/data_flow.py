from typing import Protocol


class DataFlow(Protocol):

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        ...

    def apply_raster(self, *args, **kwargs):
        ...

    def apply_mask(self, *args, **kwargs):
        ...

    def apply_image(self, *args, **kwargs):
        ...

    def apply_vector(self, *args, **kwargs):
        ...

    def apply_raster_image(self, *args, **kwargs):
        ...

    def apply_raster_h5(self, *args, **kwargs):
        ...

    def apply_sentinel_1_numpy(self, *args, **kwargs):
        ...

    def apply_sentinel_2_numpy(self, *args, **kwargs):
        ...

    def apply_mask_h5(self, *args, **kwargs):
        ...
