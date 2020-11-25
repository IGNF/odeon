# ODEON Landcover

ODEON stands for Object Delineation on Earth Observations with Neural network. It is a set of command-line tools performing semantic segmentation on remote sensing images (aerial and/or satellite) with as many layers as you wish.

## Installation

These instruction assume that you already have [conda](https://conda.io/) installed.

First, download and extract a copy of odeon from [repository](https://gitlab.com/dai-projets/odeon-landcover). Then navigate to the root of the odeon directory in a terminal and run the following:

```bash
# Install the environment
conda env create --file=environment.yml

# Activate the environment
conda activate odeon

# Install snorkel in the environment
pip install .
```

## Quickstart

Odeon toolkit is run through main command:
```bash
$ odeon
usage: odeon [-h] -c CONFIG [-v] {sample_grid,trainer}
odeon: error: the following arguments are required: tool, -c/--config
```

Each tool needs a specific JSON configuration file. Available schemas can be found in `odeon/scripts/json_defaults` folder.

More information is available in `docs` folder
