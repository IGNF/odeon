*************
Stats How To
*************

The ``stats`` script compute descriptive statistics on a dataset according to the following items:

* the bands of the images.
* the classes contained in the masks 
* the globality of the dataset

As output, the script can either generate a report file (JSON / markdown / HTML) or display directly in the terminal the obtained results.

To launch the code, type:
 ``odeon stats -c <config.json>``.

Example :

.. code-block:: console

   $ odeon stats -c ../config_files/stats_conf.json

Details on the computed statistics
=================================
Images bands statistics
-----------------------
- ``min``, ``max``: minimum and maximum pixel values for each bands. 
- ``mean``, ``std``: mean and standard deviation of the pixel distribution of each band.
- ``skewness``, ``kurtosis``: skewness and kurtosis of the pixel distribution of each band. (optional)
- histograms of pixels distributions per band with selected bins.  

Classes statistics  
------------------
- ``regu L1``: Class-Balanced Loss Based on Effective Number of
    Samples 1/frequency(i)
- ``regu L2``: Class-Balanced Loss Based on Effective Number of
    Samples 1/sqrt(frequency(i))
- ``pixel freq``: Overall share of pixels labeled with a given class.
- ``freq 5%pixel``: Share of samples with at least 5% pixels of a given class. The lesser, the more concentrated on a few samples a class is.
- ``auc``: Area under the Lorenz curve of the pixel distribution of a given class across samples. The lesser, the more concentrated on a few samples a class is. Equals pixel freq if the class is the samples are either full of or empty from the class. Equals 1 if the class is homogeneously distributed across samples.

Statistics based on the overall dataset
---------------------------------------

Global statistics are computed either with all classes or without the last class if we are not in a binary case.

- ``share multilabel``: Percentage of pixels shared by several classes
- ``avg nb class in patch``: Mean of the number of classes present in each sample 
- ``avg entropy``: Mean of the class distribution entropy for each sample. For each sample, the entropy is at least 0 if a single class is represented and at most log(C) with C the number of classes. The entropy of a sample is log(C) if every class is equally represented. The greater the entropy, the semantically richer the sample is.

Json file content
=================

**minimalist json** (the minimum configuration required to start to compute the statistics)

.. code-block:: json

 {
  "stats_setup":{
        "input_path": "/path/to/input/dataset/train.csv",
        "output_path" : "/path/to/output/file/stats.html"}
 }

.. warning::
   The extension of the output path specified in the configuration file determines the type of the output file!
   The types of output files handled are: html, json, md.
 
**full json example**

.. code-block:: json
   
   {
    "stats_setup": {
        "input_path": "/path/to/input/dataset/train.csv",
        "output_path" : "/path/to/output/file/stats.html",
        "image_bands": [0, 1, 2],
        "mask_bands": [0, 1, 3, 4]
        "data_augmentation": "rotation",
        "bins": [0, 50, 100, 150, 200, 255],
        "bit_depth": "8 bits",
        "batch_size" : 1,
        "num_workers": 1
        }
   }

**Description of JSON arguments**:
        input_path : str
            Path to .csv file describing the input dataset or a directory where the images and masks are stored.
        output_path: str
            Path where the report with the computed statistics will be created.
        image_bands: list
            List of the selected bands in the dataset images bands.
        mask_bands: list
            List of the selected bands in the dataset masks bands. (Selection of the classes)
        data_augmentation: list/str
            Data augmentation to apply in the input dataset.
        bins: list
            List of the bins to build the histograms of the image bands.
        nbr_bins: int.
            If bins is not given in input, the list of bins will be created with the nbr_bins defined here.
        get_skewness_kurtosis: bool
            Boolean to compute or not skewness and kurtosis.
        bit_depth: str
            The number of bits used to represent each pixel in an image.
        batch_size: int
            The number of image in a batch.
        num_workers: int
            Number of workers to use in the pytorch dataloader.
