import subprocess
from subprocess import CalledProcessError
import pathlib
import os
import pytest
PATH = pathlib.Path(__file__).parent.absolute()
PATH_TO_SCRIPT = os.path.join(PATH, "../odeon/sampler_grid.py")
PATH_TO_FIXTURE = os.path.join(PATH, *["fixture", "sampling"])


def run_process(*args, capture_output=True, text=True):
    try:
        return subprocess.run(args=args, capture_output=capture_output, text=text)
    except CalledProcessError as e:
        raise e


@pytest.fixture()
def run_sampler_with_good_config_1():
    args = [PATH_TO_SCRIPT] + [" ".join(["-v", "-c", os.path.join(PATH_TO_FIXTURE, "test_sampler_grid.json")])]
    run_process(args)


def test_process_no_error_warning(run_sampler_with_good_config_1):
    result = run_sampler_with_good_config_1
    print(result.stdout)
    assert 'expected out' == result.stdout


def test_process_failure():
    args = [PATH_TO_SCRIPT] + [" ".join(["-v", "-c", os.path.join(PATH_TO_FIXTURE, "test_sampler_sampler")])]
    result = subprocess.run(args=args, capture_output=True)
    print(result.returncode)
    assert 1 == int(result.returncode)
