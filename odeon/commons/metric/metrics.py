"""
Metrics tool to analyse the quality of a model's predictions.
Compute metrics, plot confusion matrices (cms) and ROC curves.
This tool handles binary and multiclass cases.
The metrics computed are in each case :

* Binary case:
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

* Multi-class case:
    - Per class: same metrics as the binary case for each class. Metrics per class and mean metrics.
    - Macro : same metrics as the binary case for the sum of all classes but without ROC/PR and calibration curve.
    - Micro : Precision, Recall, F1 Score, IoU and cm without ROC/PR and calibration curve.
"""

import os
from abc import ABC, abstractmethod
import numpy as np
from odeon import LOGGER
from odeon.commons.reports.report_factory import Report_Factory
from odeon.commons.exception import OdeonError, ErrorCodes

FIGSIZE = (8, 6)
SMOOTH = 0.000001
METRICS_NAMES = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'IoU', 'FPR']
NBR_METRICS_MICR0 = len(['Accuracy', 'IoU'])
NBR_METRICS_MACR0 = len(['Precision', 'Recall', 'F1-Score', 'IoU'])
NBR_METRICS_PER_CLASS = len(['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'IoU'])


# Dict with the default variables used to init Metrics and CLI_metrics objects.
DEFAULTS_VARS = {'output_type': 'html',
                 'class_labels': None,
                 'mask_bands': None,
                 'pred_bands': None,
                 'weights': None,
                 'threshold': 0.5,
                 'n_thresholds': 10,
                 'bins': None,
                 'n_bins': None,
                 'bit_depth': '8 bits',
                 'batch_size': 1,
                 'num_workers': 1,
                 'get_normalize': True,
                 'get_metrics_per_patch': True,
                 'get_ROC_PR_curves': True,
                 'get_ROC_PR_values': False,
                 'get_calibration_curves': True,
                 'get_hists_per_metrics': True,
                 'decimals': 2}


def get_metrics_from_obs(true_pos, false_neg, false_pos, true_neg, smooth=SMOOTH, micro=False):
    """
    Function to calculate the metrics from the observations of the number tp, fn, fp, tn of a confusion matrix.

    Parameters
    ----------
    true_pos : int
        Number of True Positive observations.
    false_neg : int
        Number of False Negative observations.
    false_pos : int
        Number of False Positive observations.
    true_neg : int
        Number of True Negative observations.

    Returns
    -------
    dict
        Dictionary containing the desired metrics.
    """
    if micro:
        oa = true_pos / (true_pos + false_pos + smooth)
        iou = true_pos / (true_pos + false_pos + false_neg + smooth)
        metrics =  {'OA': oa,
                    'IoU': iou}
    else:
        accuracy = (true_pos + true_neg) / (true_pos + false_pos + true_neg + false_neg + smooth)
        precision = true_pos / (true_pos + false_pos + smooth)
        recall = true_pos / (true_pos + false_neg + smooth)
        specificity = true_neg / (true_neg + false_pos + smooth)
        fpr = false_pos / (false_pos + true_neg + smooth)
        f1_score = (2 * true_pos) / (2 * true_pos + false_pos + false_neg + smooth)
        iou = true_pos / (true_pos + false_pos + false_neg + smooth)

        metrics =  {'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'Specificity': specificity,
                    'F1-Score': f1_score,
                    'IoU': iou,
                    'FPR': fpr}

    return metrics


class Metrics(ABC):
    """
    Abstract class Metrics to derive metrics in binary or multiclass case.
    """
    def __init__(self,
                 dataset,
                 output_path,
                 type_classifier,
                 in_prob_range,
                 output_type=DEFAULTS_VARS['output_type'],
                 class_labels=DEFAULTS_VARS['class_labels'],
                 mask_bands=DEFAULTS_VARS['mask_bands'],
                 pred_bands=DEFAULTS_VARS['pred_bands'],
                 weights=DEFAULTS_VARS['weights'],
                 threshold=DEFAULTS_VARS['threshold'],
                 n_thresholds=DEFAULTS_VARS['n_thresholds'],
                 bit_depth=DEFAULTS_VARS['bit_depth'],
                 bins=DEFAULTS_VARS['bins'],
                 n_bins=DEFAULTS_VARS['n_bins'],
                 get_normalize=DEFAULTS_VARS['get_normalize'],
                 get_metrics_per_patch=DEFAULTS_VARS['get_metrics_per_patch'],
                 get_ROC_PR_curves=DEFAULTS_VARS['get_ROC_PR_curves'],
                 get_ROC_PR_values=DEFAULTS_VARS['get_ROC_PR_values'],
                 get_calibration_curves=DEFAULTS_VARS['get_calibration_curves'],
                 get_hists_per_metrics=DEFAULTS_VARS['get_hists_per_metrics'],
                 decimals=DEFAULTS_VARS['decimals']):
        """
        Init function.
        Initialize the class attributes and create the dataframes to store the metrics.
        Once the metrics and cms are computed they are exported in an output file that can have a form json,
        markdown or html. Optionally the tool can output metrics per patch and return the result as a csv file.

        Parameters
        ----------
        dataset : MetricsDataset
            Dataset from odeon.nn.datasets which contains the masks and the predictions.
        output_path : str
            Path where the report/output data will be created.
        type_classifier : str
            String allowing to know if the classifier is of type binary or multiclass.
        in_prob_range : boolean,
            Boolean to be set to true if the values in the predictions passed as inputs are between 0 and 1.
            If not, set the parameter to false so that the tool modifies the values to be normalized between 0 and 1.
        output_type : str, optional
            Desired format for the output file. Could be json, md or html.
            A report will be created if the output type is html or md.
            If the output type is json, all the data will be exported in a dict in order
            to be easily reusable, by default html.
        class_labels : list of str, optional
            Label for each class in the dataset, by default None.
        mask_bands: list of int
            List of the selected bands in the dataset masks bands. (Selection of the classes)
        pred_bands: list of int
            List of the selected bands in the dataset preds bands. (Selection of the classes)
        weights : np.array, optional
            List of weights to balance the metrics.
            In the binary case the weights are not used in the metrics computation, by default None.
        threshold : float, optional
            Value between 0 and 1 that will be used as threshold to binarize data if they are soft.
            Use for macro, micro cms and metrics for all strategies, by default 0.5.
        n_thresholds : int, optional
            Number of thresholds used in the computation of ROC and PR, by default 10.
        bit_depth : str, optional
            The number of bits used to represent each pixel in a mask/prediction, by default '8 bits'
        bins: list of float, optional
            List of bins used for the creation of histograms.
        n_bins : int, optional
            Number of bins used in the construction of calibration curves, by default 10.
        get_normalize : bool, optional
            Boolean to know if the user wants to generate confusion matrices with normalized values, by default True
        get_metrics_per_patch : bool, optional
            Boolean to know if the user wants to compute metrics per patch and export them in a csv file.
            Metrics will be also computed if the parameter get_hists_per_metrics is True but a csv file
            won't be created, by default True
        get_ROC_PR_curves : bool, optional
            Boolean to know if the user wants to generate ROC and PR curves, by default True
        get_ROC_PR_values: bool, optional
            Boolean to know if the user wants a csv file with values used to generate ROC/PR curves, by default False
        get_calibration_curves : bool, optional
            Boolean to know if the user wants to generate calibration curves, by default True
        get_hists_per_metrics : bool, optional
            Boolean to know if the user wants to generate histogram for each metric.
            Histograms created using the parameter threshold, by default True.
        decimals: int, optional
            Number of digits after the decimal point (use for computation and display).
        """
        if os.path.exists(output_path):
            self.output_path = output_path
        else:
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Output folder ${output_path} does not exist.")

        if output_type in ['md', 'json', 'html']:
            self.output_type = output_type
        else:
            LOGGER.error('ERROR: the output file can only be in md, json, html.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input output type is incorrect.")

        self.type_classifier = type_classifier.lower()
        self.dataset = dataset
        self.nbr_class = self.dataset.nbr_class
        self.class_labels = class_labels

        if weights is None:
            self.weighted = False
            self.weights = None
        else:
            if self.type_classifier == 'binary':
                LOGGER.warning('WARNING: the parameter weigths can only be used for multiclass classifier.')
                self.weighted = False
                self.weights = None
            else:
                self.weighted = True
                self.weights = weights

        self.threshold = threshold
        self.n_thresholds = n_thresholds
        self.threshold_range = np.linspace(0.0, 1.0, self.n_thresholds)
        self.bit_depth = bit_depth
        self.define_bins(bins, n_bins)
        self.get_normalize = get_normalize
        self.get_metrics_per_patch = get_metrics_per_patch
        self.get_ROC_PR_curves = get_ROC_PR_curves
        self.get_ROC_PR_values = get_ROC_PR_values
        self.get_calibration_curves = get_calibration_curves
        self.get_hists_per_metrics = get_hists_per_metrics
        self.decimals = 2 + decimals  # Here + 2 because we wants metrics as percent between 0 and 100.
        self.metrics_names = METRICS_NAMES
        self.nbr_metrics_micro = NBR_METRICS_MICR0
        self.nbr_metrics_macro = NBR_METRICS_MACR0
        self.nbr_metrics_per_class = NBR_METRICS_PER_CLASS
        self.mask_bands = mask_bands
        self.pred_bands = pred_bands
        self.dict_export = {}

        self.depth_dict = {'keep':  1,
                           '8 bits': 255,
                           '12 bits': 4095,
                           '14 bits': 16383,
                           '16 bits': 65535}

        self.in_prob_range = in_prob_range

        if not self.get_ROC_PR_curves:
            self.threshold_range = [self.threshold]

        if self.threshold not in self.threshold_range:
            self.threshold_range = np.sort(np.append(self.threshold_range, self.threshold))

        if self.output_type == 'json':
            self.dict_export['params'] = {'class_labels': self.class_labels,
                                          'threshold': self.threshold,
                                          'threshold_range': self.threshold_range.tolist(),
                                          'bins': self.bins.tolist(),
                                          'weights': self.weights.tolist() if isinstance(self.weights, np.ndarray)
                                          else self.weights}

        self.report = Report_Factory(self)

    def __call__(self):
        """
        Create a report when the object is called.
        """
        self.run()
        self.report.create_report()

    @abstractmethod
    def run(self):
        """
        Run the methods to compute metrics.
        """
        pass

    def define_bins(self, bins, n_bins):
        """
        Create a bins list to compute probabilities histograms in functions of the
        inputs arguments n_bins, bins.

        Parameters
        ----------
        bins : list/None
            Bins to compute the histogram of the image bands.
        n_bins: int
            Number of desired bins.
        """
        if bins is None and n_bins is not None:
            bins = np.linspace(0.0, 1.0, n_bins)
        elif bins is None and n_bins is None:
            bins = np.linspace(0.0, 1.0, 11)
        else:
            assert min(bins) >= 0 and max(bins) <= 1
        self.bins = bins
        self.n_bins = len(bins)
        decimals = 2 if self.n_bins > 10 else 1

        if self.n_bins <= 20:
            self.bins_xticks = [np.round(bin_i, decimals=decimals) for bin_i in self.bins]
        else:
            self.bins_xticks = [np.round(bin_i, decimals=decimals) for bin_i in np.linspace(0.0, 1.0, 11)]

    @staticmethod
    def binarize(type_classifier, prediction, mask=None, threshold=None):
        """
        Allows the binarisation of predictions according to the type of classifier. If the classification is binary,
        the function will take in input only one prediction and will assign to each of these values either 0 or 1
        according to the threshold passed in input argument. If the classification is multiclass then the binarisation
        will be done with an argmax to return the class with the highest probability. Thus in multiclass the function
        takes in input a mask and a prediction and will return their values after applying the argmax function.

        Parameters
        ----------
        type_classifier : str
            String allowing to know if the classifier is of type binary or multiclass.
        prediction : np.array
            Prediction values.
        mask : np.array, optional
            Mask/ground truth values, by default None
        threshold : float, optional
            Threshold to binarize input data., by default None
        mask_bands: list of int
            List of the selected bands in the dataset masks bands.
        pred_bands: list of int
            List of the selected bands in the dataset preds bands.

        Returns
        -------
        np.array or Tuple(np.array)
            Transformed prediction data (with mask data if multiclass case).
        """
        pred = prediction.copy()
        output = None
        if type_classifier == 'multiclass':
            assert mask is not None
            output = (np.argmax(mask, axis=2), np.argmax(pred, axis=2))
        elif type_classifier == 'binary':
            assert threshold is not None
            pred[pred < threshold] = 0
            pred[pred >= threshold] = 1
            output = pred
        else:
            LOGGER.error('ERROR: type_classifier should be Binary or Multiclass')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input parameter 'type_classifier' is incorrect.")
        return output

    @staticmethod
    def get_confusion_matrix(truth, pred, nbr_class, revert_order=True):
        """
        Return a confusion matrix whose i-th row and j-th column entry indicates the number of samples
        with true label being i-th class and predicted label being j-th class. (example in  binary case)

                                 Predicted label
                                -----------------
                                |    1  |   0   |
                        -------------------------
                        |   1   |   TP  |   FN  |
            True label  -------------------------
                        |   0   |   FP  |   TN  |
                        -------------------------

        Parameters
        ----------
        truth : np.array
            Ground truth values.
        pred : np.array
            Prediction values.
        nbr_class: int
            Number of classes present in the input data, default None.
        revert_order: bool
            Bool to flip or not the confusion matrix.

        Returns
        -------
        np.array
            Computed confusion matrix.
        """
        assert isinstance(truth, (np.ndarray, np.generic)) and isinstance(pred, (np.ndarray, np.generic))
        class_ids = list(range(nbr_class))
        conf_mat = np.zeros([nbr_class, nbr_class], dtype=np.float64)
        for i, class_i in enumerate(class_ids):
            for j, class_j in enumerate(class_ids):
                conf_mat[i, j] = np.sum(np.logical_and(truth == class_i, pred == class_j))
        if revert_order:
            conf_mat = np.flip(conf_mat)
        return conf_mat

    def to_prob_range(self, value):
        """
        Passes values in the possible range of values for a probability i.e. between 0 and 1.
        Transformation made according to the bit depth on the input dataset.

        Parameters
        ----------
        value : number
            Input value to convert.

        Returns
        -------
        float
            Transformed data with a value between 0 and 1.
        """
        return value / self.depth_dict[self.bit_depth]

    def export_metrics_per_patch_csv(self):
        """
            Export the metrics per patch in a csv file.
        """
        path_csv = os.path.join(self.output_path, 'metrics_per_patch.csv')
        self.df_dataset.to_csv(path_csv, index=False)
