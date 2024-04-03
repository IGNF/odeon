import os
from pathlib import Path

HOME = str(Path.home())
INSTALL_PATH_VAR: str | None = os.environ.get("ODEON_INSTALL_PATH")
if INSTALL_PATH_VAR:
    ODEON_PATH: Path = Path(INSTALL_PATH_VAR, '.odeon')
else:
    ODEON_PATH = Path(HOME, '.odeon')

ODEON_ENV: Path = Path(ODEON_PATH, 'env.yaml')
