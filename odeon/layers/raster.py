from dataclasses import dataclass

import rasterio as rio


@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class Raster:
    band_indices = None
    resampling: rio.enums.Resampling = rio.enums.Resampling.nearest
