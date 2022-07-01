from dataclasses import dataclass


@dataclass
class UrbanInputFields:

    scope: str = "urbain"
    img_1: str = f"raster_2016_path"
    img_2: str = f"raster_2019_path"
    mask_1: str = f"{scope}_2016_path"
    mask_2: str = f"{scope}_2019_path"
    mask_change: str = f"{scope}_change_path"
    classes = 9


@dataclass
class NAFInputFields:

    scope: str = "naf"
    img_1: str = f"raster_2016_path"
    img_2: str = f"raster_2019_path"
    mask_1: str = f"{scope}_2016_path"
    mask_2: str = f"{scope}_2019_path"
    mask_change: str = f"{scope}_change_path"
    classes = 16


@dataclass
class EnvConf:

    root: str = "/var/data/dl/gers"
    log_path: str = "/var/data/dl/gers/log"
