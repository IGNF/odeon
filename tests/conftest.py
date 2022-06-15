import logging
import pytest
logger = logging.getLogger(__name__)
from tests import __package__


def prep_global_datadir(tmp_path_factory):

    pytest.tmp_dir = tmp_path_factory.mktemp(f"{__package__}")
    return pytest.tmp_dir


@pytest.fixture(scope="session", autouse=True)
def session_global_datadir(tmp_path_factory):
    return prep_global_datadir(tmp_path_factory)
