import logging
import os
import pathlib
from typing import Dict

import pytest

from tests import __package__

logger = logging.getLogger(__name__)


path_file = pathlib.Path(__file__).parent.resolve()


def prep_global_datadir(tmp_path_factory):

    pytest.tmp_dir = tmp_path_factory.mktemp(f"{__package__}")
    return pytest.tmp_dir


@pytest.fixture(scope="session", autouse=True)
def session_global_datadir(tmp_path_factory):
    return prep_global_datadir(tmp_path_factory)


@pytest.fixture(scope="session", autouse=True)
def path_to_test_data() -> Dict:
    return {"root_dir": os.path.join(path_file, 'test_data'),
            "zone_data": os.path.join(path_file, *['test_data', 'test_zone_data.shp']),
            "patch_data": os.path.join(path_file, *['test_data', 'test_patch_data.geojson'])}
