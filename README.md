# python ml skeleton project
generic skeleton for machine learning project with python, hydra, pytest, sphinx, github actions, etc.
with dummy functionalities!
It is mostly oriented geospatial projects

[![PyPI python](https://img.shields.io/pypi/pyversions/pmps)](https://pypi.org/project/pmps)
[![PyPI version](https://badge.fury.io/py/pmps.svg)](https://pypi.org/project/pmps)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENCE)
[![Documentation Status](https://readthedocs.org/projects/kornia/badge/?version=latest)](https://python-ml-project-skeleton.readthedocs.io/en/latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/samysung/python_ml_project_skeleton/main.svg)](https://results.pre-commit.ci/latest/github/samysung/python_ml_project_skeleton/main)
[![codecov](https://codecov.io/gh/samysung/python_ml_project_skeleton/branch/main/graph/badge.svg?token=AP5UNFJXCU)](https://codecov.io/gh/samysung/python_ml_project_skeleton)

English | [Français](README_fr-FR.md)

## Why this project?

The goal of this project is to present a standard architecture of python repository/package
including a full CiCd pipeline to document/test/deploy your project with standard methods
of 2022. It can be used as starting point for any project without reinventing the wheel.

## The code has no interest!

The code of this project is totally dummy: it makes simple
mathematics operations like addition and subtration!
The next iteration will make the opetations more interesting by
using multi-layers perceptron! It will try to add a complete example of Hydra
configuration.
<br/><br/>In a close future, it will serve as a demonstrator by the example
of a standard ML pipeline for experimentation and production

## Installation

### Install requirements
As Gdal dependencies are presents it's preferable to
install dependencis via conda before installing the package:
```bash
  git clone https://github.com/samysung/python_ml_project_skeleton
  cd python_ml_project_skeleton/packaging
  conda env create -f package_env.yml
  ```
### From pip:

  ```bash
  pip install pmps
  or pip install pmps==vx.x # for a specific version
  ```

<details>
  <summary>Other installation options</summary>

  #### From source:

  ```bash
  python setup.py install
  ```

  #### From source with symbolic links:

  ```bash
  pip install -e .
  ```

  #### From source using pip:

  ```bash
  pip install git+https://github.com/samysung/python_ml_project_skeleton
  ```
