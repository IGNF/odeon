from logging import getLogger
from typing import Dict

import albumentations as A

from odeon.core.types import STAGES_OR_VALUE
from odeon.data.dataloader_utils import DEFAULT_DATALOADER_OPTIONS
from odeon.data.stage import DataFactory

logger = getLogger(__name__)


def test_dataloader_factory_by_patch(path_to_test_data):

    input_fields: Dict = {"image": {"name": "raster_2019_path",
                                    "type": "raster", "dtype": "uint8"},
                          "mask": {"name": "naf_2019_path",
                                   "type": "mask", "encoding": "integer"}}
    root_dir: str = path_to_test_data["root_dir"]
    dataset: str = path_to_test_data["patch_data"]
    stage: STAGES_OR_VALUE = 'fit'
    input_file = dataset
    transform = None
    dataloader_options = DEFAULT_DATALOADER_OPTIONS
    input_files_has_header = False
    by_zone: bool = False
    patch_size = (256, 256)
    patch_resolution = (0.2, 0.2)
    random_window = True
    overlap = (0.0, 0.0)
    cache_dataset = False
    debug = False
    data_loader, dataset, transform, dataframe = DataFactory.build_data(input_fields=input_fields,
                                                                        input_file=input_file,
                                                                        stage=stage,
                                                                        transform=transform,
                                                                        dataloader_options=dataloader_options,
                                                                        root_dir=root_dir,
                                                                        input_files_has_header=input_files_has_header,
                                                                        by_zone=by_zone,
                                                                        patch_size=patch_size,
                                                                        patch_resolution=patch_resolution,
                                                                        random_window=random_window,
                                                                        overlap=overlap,
                                                                        cache_dataset=cache_dataset,
                                                                        debug=debug)

    n_cycle = 2
    for idx, batch in enumerate(data_loader):
        logger.info(idx)
        if int(idx) > n_cycle:
            logger.info("exit")
            break
        logger.info(f"id: {idx}, batch: {batch.keys()}")


def test_dataloader_factory_by_zone_random_window(path_to_test_data):

    input_fields: Dict = {"T-0": {"name": "T0_path", "type": "raster", "dtype": "uint8"},
                          "T-1": {"name": "t1_path", "type": "raster", "dtype": "uint8"},
                          "change_mask": {"name": "change_pat", "type": "mask", "encoding": "integer"}}
    root_dir: str = path_to_test_data["root_dir"]
    dataset: str = path_to_test_data["zone_data"]
    stage: STAGES_OR_VALUE = 'fit'
    input_file = dataset
    transform = None
    dataloader_options = DEFAULT_DATALOADER_OPTIONS
    input_files_has_header = False
    by_zone: bool = True
    patch_size = (256, 256)
    patch_resolution = (0.2, 0.2)
    random_window = True
    overlap = (0.0, 0.0)
    cache_dataset = False
    debug = False
    data_loader, dataset, transform, dataframe = DataFactory.build_data(input_fields=input_fields,
                                                                        input_file=input_file,
                                                                        stage=stage,
                                                                        transform=transform,
                                                                        dataloader_options=dataloader_options,
                                                                        root_dir=root_dir,
                                                                        input_files_has_header=input_files_has_header,
                                                                        by_zone=by_zone,
                                                                        patch_size=patch_size,
                                                                        patch_resolution=patch_resolution,
                                                                        random_window=random_window,
                                                                        overlap=overlap,
                                                                        cache_dataset=cache_dataset,
                                                                        debug=debug)

    n_cycle = 2
    for idx, batch in enumerate(data_loader):
        logger.info(idx)
        if int(idx) > n_cycle:
            logger.info("exit")
            break
        logger.info(f"id: {idx}, batch: {batch.keys()}")
        for patch_bounds, windows_bounds in zip(batch['geometry'], batch['bounds']):

            assert patch_bounds[0] <= windows_bounds[0]
            assert patch_bounds[1] <= windows_bounds[1]
            assert patch_bounds[2] >= windows_bounds[2]
            assert patch_bounds[3] >= windows_bounds[3]


def test_dataloader_factory_by_zone_center_window(path_to_test_data):

    input_fields: Dict = {"T-0": {"name": "T0_path", "type": "raster", "dtype": "uint8"},
                          "T-1": {"name": "t1_path", "type": "raster", "dtype": "uint8"},
                          "change_mask": {"name": "change_pat", "type": "mask", "encoding": "integer"}}
    root_dir: str = path_to_test_data["root_dir"]
    dataset: str = path_to_test_data["zone_data"]
    stage: STAGES_OR_VALUE = 'fit'
    input_file = dataset
    transform = None
    dataloader_options = DEFAULT_DATALOADER_OPTIONS
    input_files_has_header = False
    by_zone: bool = True
    patch_size = (256, 256)
    patch_resolution = (0.2, 0.2)
    random_window = False
    overlap = (0.0, 0.0)
    cache_dataset = False
    debug = False
    data_loader, dataset, transform, dataframe = DataFactory.build_data(input_fields=input_fields,
                                                                        input_file=input_file,
                                                                        stage=stage,
                                                                        transform=transform,
                                                                        dataloader_options=dataloader_options,
                                                                        root_dir=root_dir,
                                                                        input_files_has_header=input_files_has_header,
                                                                        by_zone=by_zone,
                                                                        patch_size=patch_size,
                                                                        patch_resolution=patch_resolution,
                                                                        random_window=random_window,
                                                                        overlap=overlap,
                                                                        cache_dataset=cache_dataset,
                                                                        debug=debug)

    n_cycle = 2
    for idx, batch in enumerate(data_loader):
        logger.info(idx)
        if int(idx) > n_cycle:
            logger.info("exit")
            break
        logger.info(f"id: {idx}, batch: {batch.keys()}")
        for patch_bounds, windows_bounds in zip(batch['geometry'], batch['bounds']):

            assert patch_bounds[0] <= windows_bounds[0]
            assert patch_bounds[1] <= windows_bounds[1]
            assert patch_bounds[2] >= windows_bounds[2]
            assert patch_bounds[3] >= windows_bounds[3]


def test_dataloader_factory_by_zone_random_window_cache_dataset(path_to_test_data):

    input_fields: Dict = {"T-0": {"name": "T0_path", "type": "raster", "dtype": "uint8"},
                          "T-1": {"name": "t1_path", "type": "raster", "dtype": "uint8"},
                          "change_mask": {"name": "change_pat", "type": "mask", "encoding": "integer"}}
    root_dir: str = path_to_test_data["root_dir"]
    stage: STAGES_OR_VALUE = 'fit'
    input_file: str = path_to_test_data["zone_data"]
    transform = [A.OneOf([A.RandomResizedCrop(height=256,
                                              width=256,
                                              scale=(0.5, 1.5),
                                              p=1.0),
                          A.RandomCrop(height=256, width=256)], p=1.0),
                 A.RandomRotate90(p=0.5),
                 A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.75),
                 A.Transpose(p=0.5)]
    dataloader_options = DEFAULT_DATALOADER_OPTIONS
    input_files_has_header = False
    by_zone: bool = True
    patch_size = (256, 256)
    patch_resolution = (0.2, 0.2)
    random_window = True
    overlap = (0.0, 0.0)
    cache_dataset = True
    debug = False
    data_loader, dataset, transform, dataframe = DataFactory.build_data(input_fields=input_fields,
                                                                        input_file=input_file,
                                                                        stage=stage,
                                                                        transform=transform,
                                                                        dataloader_options=dataloader_options,
                                                                        root_dir=root_dir,
                                                                        input_files_has_header=input_files_has_header,
                                                                        by_zone=by_zone,
                                                                        patch_size=patch_size,
                                                                        patch_resolution=patch_resolution,
                                                                        random_window=random_window,
                                                                        overlap=overlap,
                                                                        cache_dataset=cache_dataset,
                                                                        debug=debug)

    n_cycle = 2
    for i in range(n_cycle):
        for idx, batch in enumerate(data_loader):
            logger.info(idx)
            if int(idx) > n_cycle:
                logger.info("exit")
                break
            logger.info(f"id: {idx}, batch: {batch.keys()}")
            for patch_bounds, windows_bounds in zip(batch['geometry'], batch['bounds']):

                assert patch_bounds[0] <= windows_bounds[0]
                assert patch_bounds[1] <= windows_bounds[1]
                assert patch_bounds[2] >= windows_bounds[2]
                assert patch_bounds[3] >= windows_bounds[3]
    logger.info(dataset.preprocess.cache_dataset)
    logger.info(dataset.preprocess.cache)
