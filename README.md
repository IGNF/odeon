[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# ODEON Landcover

ODEON stands for Object Delineation on Earth Observations with Neural network.
It is a set of command-line tools performing semantic segmentation on remote
sensing images (aerial and/or satellite) with as many layers as you wish.

## Installation

These instructions assume that you already have [conda](https://conda.io/) installed.

First, download and extract a copy of odeon from [repository](https://gitlab.com/StephanePEILLET/odeon-landcover).
Then navigate to the root of the odeon directory in a terminal and run the following:

```bash
# Clone repository
git clone git@gitlab.com:StephanePEILLET/odeon-landcover.git
or
git clone https://gitlab.com/StephanePEILLET/odeon-landcover.git
or
download a release at https://gitlab.com/StephanePEILLET/odeon-landcover/-/releases

# Go to the root project folder
cd odeon-landcover

# Install the environment
conda env create --file=environment.yml

# Activate the environment
conda activate odeon

# Install snorkel in the environment
pip install .
```
## Documentation
You can find the documentation of the project at [https://stephanepeillet.gitlab.io/odeon-landcover/](https://stephanepeillet.gitlab.io/odeon-landcover)

## Quickstart

Odeon toolkit is run through main command:
```bash
$ odeon
usage: odeon [-h] -c CONFIG [-v] {sample_grid,trainer}
odeon: error: the following arguments are required: tool, -c/--config
```

Each tool needs a specific JSON configuration file. Available schemas can be found in `odeon/cli/json_defaults` folder.

More information is available in `docs` folder
