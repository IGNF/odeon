***************
Metrics how-to
***************

``metrics`` script compute metrics in order to analyse the quality of a model's predictions.
The tool compute metrics, plot confusion matrices (cms) and ROC curves.
This tool handles binary and multiclass cases with prediction in 'soft' or 'hard' probabilities.

To launch the code, type:
 ``odeon metrics -c <config.json>``.

Example :

.. code-block:: console

   $ odeon metrics -c ../config_files/metrics_conf.json


The metrics computed
====================

Binary case
-----------

    - Confusion matrix (cm)
    - (optional) normalized by classes cm.
    - Accuracy
    - Precision
    - Recall
    - Specificity
    - F1 Score
    - IoU
    - ROC and PR curves
    - AUC Score for ROC/PR curves
    - Calibration Curve
    - Histogram for each metric

Multi-class case
----------------
    - Per class: same metrics as the binary case for each class. Metrics per class and mean metrics.
    - Macro : same metrics as the binary case for the sum of all classes but without ROC/PR and calibration curve.
    - Micro : Precision, Recall, F1 Score, IoU and cm without ROC/PR and calibration curve.

**Binary case**
===============

Confusion Matrix
----------------

The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:

.. figure:: assets/metrics/metrics_cm.png
   :align: center
   :figclass: align-center

Main metrics
------------

The following metrics are commonly used to assess the performance of classification models:

.. figure:: assets/metrics/metrics_main_formula.png
   :align: center
   :figclass: align-center

ROC Curve
---------

The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:

.. figure:: assets/metrics/metrics_ROC_def.png
   :align: center
   :figclass: align-center

The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:

.. figure:: assets/metrics/metrics_ROC_curve.png
   :align: center
   :figclass: align-center

PR Curve
--------

The precision-recall curve shows the tradeoff between precision and recall for different threshold. 
Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced.

Example:

.. figure:: assets/metrics/metrics_pr_curve.png
   :align: center
   :figclass: align-center

**Multiclass case**
===================

**Macro Strategy**
------------------

Macro strategy consists in looking at the performance of a model from a more global point of view.
To do so, each class is first treated as in a binary case (1 vs all) in order to produce a confusion matrix for each class.
The confusion matrices are then added together to form a single matrix which will be our macro confusion matrix.

.. note::
    The sum of the classes can be done in a pondered way by entering weights argument in the configuration file.
    These weights can be used to rebalance the importance of a class on a metric, or even by setting the weights to 0 for a class,
    this class will not be taken into account for the calculation of macro and mean metrics. 

Example:

.. figure:: assets/metrics/metrics_macro_df.png
   :align: center
   :figclass: align-center

**Micro Strategy**
------------------

The micro strategy provides a global but more accurate view of the performance of a model.
The quality of a prediction will not be judged by class but by looking at the whole number of TP, FN and FP made by the model.

Example of micro confusion matrix with 3 classes:

.. list-table:: Confusion Matrice for micro strategy
   :widths: 20 20 20 20

   * - 
     - A
     - B
     - C
   * - A
     - TP
     - FN
     - FN
   * - B
     - FP
     - TP
     - FN
   * - C
     - FP
     - FP
     - TP

Example:

.. figure:: assets/metrics/metrics_cm_micro.png
   :align: center
   :figclass: align-center

.. figure:: assets/metrics/metrics_micro_df.png
   :align: center
   :figclass: align-center

.. note::
    It is possible to have a normalized confusion matrix per class as in the image above right.
    This allows to see for a class the distribution of these predictions. In order to do this you need to use the `get_normalize`.

**Per class strategy**
----------------------

The class strategy is the even more precise view but only looks at the performance of each class one by one and independently. 
Example of a confusion matrix for a class in  a multiclass case, here class A.

.. list-table:: Confusion Matrice for class A
   :widths: 20 20 20 20

   * - 
     - A
     - B
     - C
   * - A
     - TP
     - FN
     - FN
   * - B
     - FP
     - TN
     - TN
   * - C
     - FP
     - TN
     - TN

Example:

.. figure:: assets/metrics/metrics_classes_df.png
   :align: center
   :figclass: align-center


Json file content
=================

.. details:: **minimalist json** (the minimum configuration required to start to compute the statistics)
    .. code-block:: json
        {
            "metrics_setup": {
                "mask_path": "/path/to/intput/folder/msk",
                "pred_path": "/path/to/input/folder/pred",
                "output_path": "/path/to/output/folder/",
                "type_classifier": "binary"
                }
        }
 
.. warning::
   By default, the format of the ouput fil will be "html".

.. details:: **full json example**
    .. code-block:: json
        {
            "metrics_setup": {
                "mask_path": "/path/to/intput/folder/msk",
                "pred_path": "/path/to/input/folder/pred",
                "output_path": "/path/to/output/folder/",
                "type_classifier": "multiclass",
                "weights": [0.3, 0.5, 0.0, 0.0, 0.9, 0.1, 0.1],
                "class_labels": ["batiments", "route", "ligneux", "herbacé", "eau", "mineraux", "piscines"],
                "threshold": 0.6,
                "threshold_range": [0.45, ,0.5, 0.55, 0.6, 0.65, 0.7],
                "bit_depth": "8 bits",
                "nb_calibration_bins": 10,
                "get_normalize": true,
                "get_metrics_per_patch": true,
                "get_ROC_PR_curves": true,
                "get_calibration_curves": false,
                "get_hists_per_metrics": false
            }
        }

**Description of JSON arguments**
- ``mask_path`` : str, required
    Path to the folder containing the masks.
- ``pred_path`` : str, required
    Path to the folder containing the predictions.
- ``output_path`` : str, required
    Path where the report/output data will be created.
- ``type_classifier`` : str, required
    String allowing to know if the classifier is of type binary or multiclass.
- ``output_type`` : str, optional
    Desired format for the output file. Could be json, md or html.
    A report will be created if the output type is html or md.
    If the output type is json, all the data will be exported in a dict in order
    to be easily reusable, by default html.
- ``class_labels`` : list of str, optional
    Label for each class in the dataset.
    If None the labels of the classes will be of type:  0 and 1 by default None
- ``weights`` : list of number, optional
    List of weights to balance the metrics.
    In the binary case the weights are not used in the metrics computation, by default None.
- ``threshold`` : float, optional
    Value between 0 and 1 that will be used as threshold to binarize data if they are soft.
    Use for macro, micro cms and metrics for all strategies, by default 0.5.
- ``threshold_range`` : list of float, optional
    List of values that will be used as a threshold when calculating the ROC and PR curves,
    by default np.arange(0.1, 1.1, 0.1).
- ``bit_depth`` : str, optional
    The number of bits used to represent each pixel in a mask/prediction, by default '8 bits'
- ``nb_calibration_bins`` : int, optional
    Number of bins used in the construction of calibration curves, by default 10.
- ``get_normalize`` : bool, optional
    Boolean to know if the user wants to generate confusion matrices with normalized values, by default True
- ``get_metrics_per_patch`` : bool, optional
    Boolean to know if the user wants to compute metrics per patch and export them in a csv file.
    Metrics will be also computed if the parameter get_hists_per_metrics is True but a csv file
    won't be created, by default True
- ``get_ROC_PR_curves`` : bool, optional
    Boolean to know if the user wants to generate ROC and PR curves, by default True
- ``get_calibration_curves`` : bool, optional
    Boolean to know if the user wants to generate calibration curves, by default True
- ``get_hists_per_metrics`` : bool, optional
    Boolean to know if the user wants to generate histogram for each metric.
    Histograms created using the parameter threshold, by default True.