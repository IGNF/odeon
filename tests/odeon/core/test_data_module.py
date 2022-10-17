from logging import getLogger
from typing import Dict

from odeon.data.data_module import Input

logger = getLogger(__name__)


def test_patch_data_module_creation(path_to_test_data):

    input_fields: Dict = {"image": {"name": "raster_2019_path", "type": "raster", "dtype": "uint8"},
                          "mask": {"name": "naf_2019_path", "type": "mask", "encoding": "integer"}}

    root_dir: str = path_to_test_data["root_dir"]
    dataset: str = path_to_test_data["patch_data"]
    data_module = Input(input_fields=input_fields,
                        root_dir=root_dir,
                        input_fit_file=dataset,
                        input_validate_file=dataset)
    train_data_loader = data_module.train_dataloader()
    # logger.info(data_module)
    # logger.info(train_data_loader)
    n_cycle = 2
    for idx, batch in enumerate(train_data_loader):
        logger.info(idx)
        if int(idx) > n_cycle:
            logger.info("exit")
            break
        logger.info(f"id: {idx}, batch: {batch.keys()}")


def test_zone_data_module_creation(path_to_test_data):

    input_fields: Dict = {"T-0": {"name": "T0_path", "type": "raster", "dtype": "uint8"},
                          "T-1": {"name": "t1_path", "type": "raster", "dtype": "uint8"},
                          "change_mask": {"name": "change_pat", "type": "mask", "encoding": "integer"}}

    root_dir: str = path_to_test_data["root_dir"]
    dataset: str = path_to_test_data["zone_data"]

    data_module = Input(input_fields=input_fields,
                        root_dir=root_dir,
                        input_fit_file=dataset,
                        input_validate_file=dataset,
                        by_zone="all")

    train_data_loader = data_module.train_dataloader()

    # logger.info(data_module)
    # logger.info(train_data_loader)
    n_cycle = 2
    for idx, batch in enumerate(train_data_loader):
        logger.info(idx)
        if idx > n_cycle:
            logger.info("exit")
            break
        logger.info(f"id: {idx}, batch: {batch.keys()}, window_bounds: {batch['bounds']}, bounds: {batch['geometry']}")
        for patch_bounds, windows_bounds in zip(batch['geometry'], batch['bounds']):

            assert patch_bounds[0] <= windows_bounds[0]
            assert patch_bounds[1] <= windows_bounds[1]
            assert patch_bounds[2] >= windows_bounds[2]
            assert patch_bounds[3] >= windows_bounds[3]


def test_zone_data_module_creation_with_cached_dataset(path_to_test_data):

    input_fields: Dict = {"T-0": {"name": "T0_path", "type": "raster", "dtype": "uint8"},
                          "T-1": {"name": "t1_path", "type": "raster", "dtype": "uint8"},
                          "change_mask": {"name": "change_pat", "type": "mask", "encoding": "integer"}}

    root_dir: str = path_to_test_data["root_dir"]
    dataset: str = path_to_test_data["zone_data"]
    data_module = Input(input_fields=input_fields,
                        root_dir=root_dir,
                        input_fit_file=dataset,
                        input_validate_file=dataset,
                        by_zone="all",
                        cache_dataset="all")

    train_data_loader = data_module.train_dataloader()

    # logger.info(data_module)
    # logger.info(train_data_loader)
    n_cycle = 2
    for idx, batch in enumerate(train_data_loader):

        if idx > n_cycle:

            logger.info("exit")
            break

        logger.info(f"id: {idx}, batch: {batch.keys()}, window_bounds: {batch['bounds']}, bounds: {batch['geometry']}")
        for patch_bounds, windows_bounds in zip(batch['geometry'], batch['bounds']):

            assert patch_bounds[0] <= windows_bounds[0]
            assert patch_bounds[1] <= windows_bounds[1]
            assert patch_bounds[2] >= windows_bounds[2]
            assert patch_bounds[3] >= windows_bounds[3]


def test_zone_data_module_creation_for_inference(path_to_test_data):

    input_fields: Dict = {"T-0": {"name": "T0_path", "type": "raster", "dtype": "uint8"},
                          "T-1": {"name": "t1_path", "type": "raster", "dtype": "uint8"},
                          "change_mask": {"name": "change_pat", "type": "mask", "encoding": "integer"}}

    root_dir: str = path_to_test_data["root_dir"]
    dataset: str = path_to_test_data["zone_data"]
    data_module = Input(input_fields=input_fields,
                        root_dir=root_dir,
                        input_fit_file=dataset,
                        input_validate_file=dataset,
                        by_zone="all",
                        cache_dataset="all")

    data_loader = data_module.val_dataloader()

    # logger.info(data_module)
    # logger.info(train_data_loader)
    n_cycle = 2
    for idx, batch in enumerate(data_loader):

        if idx > n_cycle:

            logger.info("exit")
            break

        logger.info(f"id: {idx}, batch: {batch.keys()}, window_bounds: {batch['bounds']}, bounds: {batch['geometry']}")
        for patch_bounds, windows_bounds in zip(batch['geometry'], batch['bounds']):

            assert patch_bounds[0] <= windows_bounds[0]
            assert patch_bounds[1] <= windows_bounds[1]
            assert patch_bounds[2] >= windows_bounds[2]
            assert patch_bounds[3] >= windows_bounds[3]
