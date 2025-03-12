from logging import getLogger
from pathlib import Path

from setuptools import find_packages, setup  # type: ignore #

logger = getLogger(__name__)

try:
    with open(str(Path(__file__).parent / "README.md"), "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    logger.error("Could not find the README.md file during setup install")
try:
    with open(str(Path(__file__).parent / "VERSION"), "r") as fh:
        version = fh.read().strip()
except FileNotFoundError:
    logger.error("Could not find the VERSION file during setup install")
setup(
    name="odeon",
    version=version,  # Replace with your desired version or load from file as before
    author="samy khelifi",
    author_email="samy.khelifi@ign.fr",
    description="odeon is a framework for development of machine learning pipeline in remote sensing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IGNF/odeon",
    project_urls={"Bug Tracker": "https://github.com/IGNF/odeon/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    packages=find_packages("."),
    python_requires=">=3.10",
    install_requires=[
        "tqdm",
        "numpy>=1.17.2",
        "omegaconf>=2.1",
        "jsonargparse",
        # Geo Dependencies
        "geopandas>=0.10",
        "fiona>=1.8",
        "pyproj>=2.2",
        "rasterio>=1.0.20",
        "rtree>=1",
        "shapely>=1.3",
        "pyogrio",
        # Machine Learning / Dataviz / Computer Vision
        "matplotlib",
        "scikit-learn>=0.21",
        "seaborn",
        "matplotlib-base>=3.3,<4",
        "pillow",
        "opencv-python-headless>=4.7.0",
        # Deep Learning / Pytorch ecosystem
        "torch>=2.0.0",
        "torchvision",
        "pytorch-lightning",
        "segmentation-models-pytorch",
        "timm",
        "torchmetrics",
        "tensorboard",
        "tensorboardx",
        "albumentations",
        "kornia>=0.7",
        "einops",
    ],
    entry_points={"console_scripts": ["odeon = odeon.main:main"]},
    package_data={"odeon": ["../VERSION", "../README.md", "../LICENSE", "../README_fr-FR.md"]},
    include_package_data=True,
)
