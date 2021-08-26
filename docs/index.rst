.. odeon-landcover documentation master file, created by
   sphinx-quickstart on Thu Nov 19 20:53:21 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
*******************************
Odeon-landcover's documentation
*******************************

ODEON stands for Object Delineation on Earth Observations with Neural network.
It is a set of command-line tools performing semantic segmentation on remote
sensing images (aerial and/or satellite) with as many layers as you wish.

Installation
============

These instruction assume that you already have `conda <https://conda.io/>`_ installed.

First, download and extract a copy of odeon from `repository <https://gitlab.com/dai-projets/odeon-landcover>`_
Then navigate to the root of the odeon directory in a terminal and run the following:

**Install and activate the environment**

.. code-block:: console

   $ clone repository https://gitlab.com/dai-projets/odeon-landcover
   $ cd odeon-landcover
   $ conda env create --file=environment.yml
   $ conda activate odeon
   $ pip install .

In order to use cuda and NVIDIA devices, cudatoolkit must be installed too.


Quickstart
==========

Odeon toolkit is run through main command:

.. code-block:: console

   $ odeon
     usage: odeon [-h] -c CONFIG [-v] {sample_grid,trainer}
     odeon: error: the following arguments are required: tool, -c/--config

Each tool needs a specific JSON configuration file.
Available schemas can be found in `odeon/scripts/json_defaults` folder.

More information could be found in the documentation

.. toctree::
    :caption: User Guide
    :maxdepth: 1

    Grid sampling <sample_grid.rst>
    Systematic sampling <sample_sys.rst>
    Dataset Generation <generate.rst>
    Stats <stats.rst>
    Training <train.rst>
    Detection <detect.rst>
    Metrics <metrics.rst>
    
.. toctree::
    :caption: Advanced Feature
    :maxdepth: 1
    
    Setup <setup.rst>
    
.. toctree::
    :caption: API Reference
    :maxdepth: 1
    
    Tools <tools.rst>
    Models <models.rst>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
