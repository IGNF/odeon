# ODEON

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)


English | [Francais](./README_fr-FR.md)

## Why this name
ODEON stands for Object Delineation on Earth Observations with Neural network.

## What is the purpose of this library?
It used to be a set of command line tool for semantic segmentation,
it is now pivoting to an agnostic framework for deep learning applied to
GIS industry

## Installation
The new version is still in high development phase, but you can still
use the legacy version

### Installation requirements
As Gdal dependencies are presents we recommend to
install the dependencies via conda/mamba before installing the package:
#### Legacy version
```bash
  git clone -b odeon-legacy git@github.com:IGNF/odeon.git
  cd cd odeon
  conda(or mamba) env create -f package_env.yml
  pip install -e .
```
#### New version
```bash
  git clone git:odeon-legacy git@github.com:IGNF/odeon.git
  cd cd odeon/packaging
  conda(or mamba) env create -f package_env.yaml
  pip install -e .
  ```
