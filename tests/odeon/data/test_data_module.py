from logging import getLogger

from odeon.data.data_module import Input

logger = getLogger(__name__)


def test_data_module_only_one_fit_minimal_config(path_to_test_data):
    root_dir: str = path_to_test_data["root_dir"]
    dataset: str = path_to_test_data["patch_data"]
    fit_params = {
        'input_fields': {"image": {"name": "raster_2019_path",
                                   "type": "raster", "dtype": "uint8"},
                         "mask": {"name": "naf_2019_path", "type": "mask", "encoding": "integer"}},
        'input_file': dataset,
        'root_dir': root_dir
    }

    data_module = Input(fit_params=fit_params)
    data_loader = data_module.train_dataloader()
    n_cycle = 2
    for i in range(n_cycle):
        for idx, batch in enumerate(data_loader):
            logger.info(idx)
            if int(idx) > n_cycle:
                logger.info("exit")
                break
            logger.info(f"id: {idx}, batch: {batch.keys()}")


def test_data_module_multi_fit(path_to_test_data):
    root_dir: str = path_to_test_data["root_dir"]
    dataset: str = path_to_test_data["patch_data"]
    fit_params = {
        'input_fields': {"image": {"name": "raster_2019_path",
                                   "type": "raster", "dtype": "uint8"},
                         "mask": {"name": "naf_2019_path",
                                  "type": "mask", "encoding": "integer"}},
        'input_file': dataset,
        'root_dir': root_dir
    }

    data_module = Input(fit_params=[fit_params, fit_params])
    data_loader = data_module.train_dataloader()
    logger.info(f'data lodaders: {data_loader}')
    n_cycle = 2
    for i in range(n_cycle):
        for idx, (batch1, batch2) in enumerate(zip(data_loader['fit-1'], data_loader['fit-2'])):
            logger.info(idx)
            if int(idx) > n_cycle:
                logger.info("exit")
                break
            logger.info(f"id: {idx}, batch: {batch1.keys()}")
            logger.info(f"id: {idx}, batch: {batch2.keys()}")


def test_data_module_multi_stage(path_to_test_data):
    ...


def test_data_module_multi_fit_multi_stage(path_to_test_data):

    root_dir: str = path_to_test_data["root_dir"]
    dataset: str = path_to_test_data["patch_data"]
    fit_params = {
        'input_fields': {"image": {"name": "raster_2019_path",
                                   "type": "raster", "dtype": "uint8"},
                         "mask": {"name": "naf_2019_path",
                                  "type": "mask", "encoding": "integer"}},
        'input_file': dataset,
        'root_dir': root_dir
    }
    predict_params = test_params = validate_params = fit_params

    data_module = Input(fit_params=[fit_params, fit_params],
                        validate_params=validate_params,
                        test_params=test_params,
                        predict_params=predict_params)
    data_loader = data_module.train_dataloader()
    logger.info(f'data lodaders: {data_loader}')
    n_cycle = 2
    for i in range(n_cycle):
        for idx, (batch1, batch2) in enumerate(zip(data_loader['fit-1'], data_loader['fit-2'])):
            logger.info(idx)
            if int(idx) > n_cycle:
                logger.info("exit")
                break
            logger.info(f"id: {idx}, batch: {batch1.keys()}")
            logger.info(f"id: {idx}, batch: {batch2.keys()}")

    # validate dataloader
    data_loader = data_module.val_dataloader()
    n_cycle = 2
    for i in range(n_cycle):
        for idx, batch in enumerate(data_loader):
            logger.info(idx)
            if int(idx) > n_cycle:
                logger.info("exit")
                break
            logger.info(f"id: {idx}, batch: {batch.keys()}")

    # test dataloader
    data_loader = data_module.test_dataloader()
    n_cycle = 2
    for i in range(n_cycle):
        for idx, batch in enumerate(data_loader):
            logger.info(idx)
            if int(idx) > n_cycle:
                logger.info("exit")
                break
            logger.info(f"id: {idx}, batch: {batch.keys()}")

    # predict dataloader
    data_loader = data_module.predict_dataloader()
    n_cycle = 2
    for i in range(n_cycle):
        for idx, batch in enumerate(data_loader):
            logger.info(idx)
            if int(idx) > n_cycle:
                logger.info("exit")
                break
            logger.info(f"id: {idx}, batch: {batch.keys()}")
