import pytorch_lightning as pl


def is_pytorch_lightning_2():
    """
    Check if PyTorch Lightning version is 2.0 or later.

    Returns
    -------
    bool
        True if PyTorch Lightning version is 2.0 or later, False otherwise.

    Raises
    ------
    ImportError
        If PyTorch Lightning is not installed.
    """
    try:
        # Get the PyTorch Lightning version
        lightning_version = pl.__version__
        major_version = int(lightning_version.split('.')[0])
        # Check if the major version is 2 or greater
        if major_version >= 2:
            return True
        else:
            return False
    except ImportError:
        # PyTorch Lightning is not installed
        raise ImportError("PyTorch Lightning is not installed.")


def is_pytorch_lightning_superior_21():
    """
    Check if PyTorch Lightning version is greater than 2.1.

    Returns
    -------
    bool
        True if PyTorch Lightning version is greater than 2.1, False otherwise.

    Raises
    ------
    ImportError
        If PyTorch Lightning is not installed.
    """
    try:
        # Get the PyTorch Lightning version
        lightning_version = pl.__version__
        major_version, minor_version = map(int, lightning_version.split('.')[:2])
        # Check if the version is greater than 2.1
        if major_version > 2 or (major_version == 2 and minor_version > 1):
            return True
        else:
            return False
    except ImportError:
        # PyTorch Lightning is not installed
        raise ImportError("PyTorch Lightning is not installed.")


PYTORCH_LIGHTNING_SUP_2 = is_pytorch_lightning_2()
PYTORCH_LIGHTNING_SUP_21 = is_pytorch_lightning_superior_21()
