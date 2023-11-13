from pathlib import Path

HOME = str(Path.home())
ODEON_PATH: Path = Path(HOME, '.odeon')
ODEON_ENV: Path = Path(ODEON_PATH, 'env.yaml')
