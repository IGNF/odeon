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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
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
                 'threshold': 0.5,
                 'n_thresholds': 10,
                 'weights': None,
                 'bins': None,
                 'n_bins': None,
                 'bit_depth': '8 bits',
                 'batch_size': 1,
                 'num_workers': 1,
                 'get_normalize': True,
                 'get_metrics_per_patch': True,
                 'get_ROC_PR_curves': True,
                 'get_calibration_curves': True,
                 'get_hists_per_metrics': True}


class Metrics(ABC):

    def __init__(self,
                 dataset,
                 output_path,
                 type_classifier,
                 in_prob_range,
                 class_labels=None,
                 output_type=DEFAULTS_VARS['output_type'],
                 weights=DEFAULTS_VARS['weights'],
                 threshold=DEFAULTS_VARS['threshold'],
                 n_thresholds=DEFAULTS_VARS['n_thresholds'],
                 bit_depth=DEFAULTS_VARS['bit_depth'],
                 bins=DEFAULTS_VARS['bins'],
                 n_bins=DEFAULTS_VARS['n_bins'],
                 get_normalize=DEFAULTS_VARS['get_normalize'],
                 get_metrics_per_patch=DEFAULTS_VARS['get_metrics_per_patch'],
                 get_ROC_PR_curves=DEFAULTS_VARS['get_ROC_PR_curves'],
                 get_calibration_curves=DEFAULTS_VARS['get_calibration_curves'],
                 get_hists_per_metrics=DEFAULTS_VARS['get_hists_per_metrics']):
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
        weights : list of number, optional
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
        get_calibration_curves : bool, optional
            Boolean to know if the user wants to generate calibration curves, by default True
        get_hists_per_metrics : bool, optional
            Boolean to know if the user wants to generate histogram for each metric.
            Histograms created using the parameter threshold, by default True.
        """
        if not os.path.exists(output_path):
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Output folder ${output_path} does not exist.")
        else:
            self.output_path = output_path

        if output_type in ['md', 'json', 'html']:
            self.output_type = output_type
        else:
            LOGGER.error('ERROR: the output file can only be in md, json, html.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input output type is incorrect.")

        self.type_classifier = type_classifier.lower()
        self.dataset = dataset
        self.nbr_class = self.dataset.nbr_class

        if class_labels is not None and all(class_labels):
            self.class_labels = class_labels
        else:
            self.class_labels = [f'class {i + 1}' for i in range(self.nbr_class)]

        if weights is None:
            self.weights = np.ones(self.nbr_class)
            self.weighted = False
        elif weights is not None and self.type_classifier == 'binary':
            LOGGER.warning('WARNING: the parameter weigths can only be used for multiclass classifier.')
            self.weights = weights
            self.weighted = False
        elif len(weights) != self.nbr_class:
            LOGGER.error('ERROR: parameter weigths should have a number of values equal to the number of classes.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input parameter weigths is incorrect.")
        else:
            self.weights = weights
            self.weighted = True

        self.class_ids = np.arange(self.nbr_class)  # Each class is identified by a number
        self.threshold = threshold
        self.n_thresholds = n_thresholds
        self.threshold_range = np.linspace(0.0, 1.0, self.n_thresholds)
        self.bit_depth = bit_depth
        self.define_bins(bins, n_bins)
        self.get_normalize = get_normalize
        self.get_metrics_per_patch = get_metrics_per_patch
        self.get_ROC_PR_curves = get_ROC_PR_curves
        self.get_calibration_curves = get_calibration_curves
        self.get_hists_per_metrics = get_hists_per_metrics
        self.metrics_names = METRICS_NAMES
        self.nbr_metrics_micro = NBR_METRICS_MICR0
        self.nbr_metrics_macro = NBR_METRICS_MACR0
        self.nbr_metrics_per_class = NBR_METRICS_PER_CLASS

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
            self.dict_export = {}
            self.dict_export['params'] = {'class_labels': self.class_labels,
                                          'threshold': self.threshold,
                                          'threshold_range': self.threshold_range.tolist(),
                                          'bins': self.bins.tolist(),
                                          'weights': self.weights}
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
            bins = bins
        self.bins = bins
        self.n_bins = len(bins)
        decimals = 2 if self.n_bins > 10 else 1

        if self.n_bins <= 20:
            self.bins_xticks = [np.round(bin_i, decimals=decimals) for bin_i in self.bins]
        else:
            self.bins_xticks = [np.round(bin_i, decimals=decimals) for bin_i in np.linspace(0.0, 1.0, 11)]

    def binarize(self, type_classifier, prediction, mask=None, threshold=None):
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

        Returns
        -------
        np.array
            Transformed prediction data.
        """
        pred = prediction.copy()
        if not self.in_prob_range:
            pred = self.to_prob_range(pred)
        if type_classifier == 'multiclass':
            assert mask is not None
            return np.argmax(mask, axis=2), np.argmax(pred, axis=2)
        elif type_classifier == 'binary':
            assert threshold is not None
            pred[pred < threshold] = 0
            pred[pred >= threshold] = 1
            return pred
        else:
            LOGGER.error('ERROR: type_classifier should be Binary or Multiclass')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input parameter 'type_classifier' is incorrect.")

    def get_confusion_matrix(self, truth, pred, nbr_class=None, revert_order=True):
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

        Returns
        -------
        np.array
            Computed confusion matrix.
        """
        assert isinstance(truth, (np.ndarray, np.generic)) and isinstance(pred, (np.ndarray, np.generic))
        if nbr_class is None:
            nbr_class = self.nbr_class
            class_ids = self.class_ids
        else:
            class_ids = list(range(nbr_class))

        cm = np.zeros([nbr_class, nbr_class], dtype=np.float64)
        for i, class_i in enumerate(class_ids):
            for j, class_j in enumerate(class_ids):
                cm[i, j] = np.sum(np.logical_and(truth == class_i, pred == class_j))
        if revert_order:
            return np.flip(cm)
        else:
            return cm

    @staticmethod
    def get_metrics_from_obs(tp, fn, fp, tn, smooth=SMOOTH):
        """
        Function to calculate the metrics from the observations of the number tp, fn, fp, tn of a confusion matrix.

        Parameters
        ----------
        tp : int
            Number of True Positive observations.
        fn : int
            Number of False Negative observations.
        fp : int
            Number of False Positive observations.
        tn : int
            Number of True Negative observations.

        Returns
        -------
        dict
            Dictionary containing the desired metrics.
        """
        accuracy = (tp + tn) / (tp + fp + tn + fn + smooth)
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        specificity = tn / (tn + fp + smooth)
        fpr = fp / (fp + tn + smooth)
        f1_score = (2 * tp) / (2 * tp + fp + fn + smooth)
        iou = tp / (tp + fp + fn + smooth)

        return {'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'F1-Score': f1_score,
                'IoU': iou,
                'FPR': fpr}

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

    @staticmethod
    def heatmap(data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.
        Code from : https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        ax.set_ylabel('Actual Class')
        ax.set_xlabel('Predicted Class')
        ax.xaxis.set_label_position('top')

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    @staticmethod
    def annotate_heatmap(im, data=None, valfmt="{x:.3f}",
                         textcolors=("black", "white"), threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center", verticalalignment="center")
        kw.update(textkw)

        # # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        if isinstance(valfmt, (np.ndarray, np.generic)):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                    if valfmt[i, j] == 'nodata':
                        text = im.axes.text(j, i, valfmt[i, j], **kw)
                    else:
                        decimals = 1 if data[i, j] >= 1 else 3
                        text = im.axes.text(j, i,
                                            str(np.round(data[i, j] / valfmt[i, j][0] if data[i, j] != 0 else 0,
                                                         decimals))
                                            + valfmt[i, j][1],
                                            **kw)
                    texts.append(text)
        else:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                    text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                    texts.append(text)
        return texts

    @staticmethod
    def get_cm_val_fmt(cm, mark_no_data=False):
        """
        Function allowing to obtain a matrix containing the elements necessary to format each cell of a confusion matrix
        so that the number of observations can be entered in a cell. Each element of the matrix consist of tuple with a
        number to divide the value of the cm case and character to show the unit.
        ex: cm value = 3000 -> fmt (1000, 'k') -> '3k'.

        Parameters
        ----------
        cm : np.array
            Confusion matrix with float values to format.

        Returns
        -------
        np.array
            Matrix with elements to format the cm.
        """

        def find_val_fmt(value):
            """Return format element for one value.

            Parameters
            ----------
            value : float
                value to transform.

            Returns
            -------
            Tuple(int, str)
                Value to divide the input value, character to know in which unit is the input value.
            """
            length_dict = {0: (10**0, ''),
                           3: (10**3, 'k'),
                           6: (10**6, 'm'),
                           9: (10**9, 'g'),
                           12: (10**12, 't'),
                           15: (10**15, 'p')}
            divider, unit_char = None, None
            for i, length in enumerate(length_dict):
                number = str(value).split('.')[0]
                if len(number) < length + 1:
                    divider = length_dict[list(length_dict)[i - 1]][0]
                    unit_char = length_dict[list(length_dict)[i - 1]][1]
                    break
                elif len(number) == length + 1:
                    divider = length_dict[length][0]
                    unit_char = length_dict[length][1]
                    break
                elif i == len(length_dict) - 1:
                    divider = length_dict[list(length_dict)[i]][0]
                    unit_char = length_dict[list(length_dict)[i]][1]
            return (divider, unit_char)

        cm_val_fmt = np.zeros_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            if mark_no_data and all(np.equal(cm[i], 0)):
                cm_val_fmt[i] = ['nodata' for _ in range(cm.shape[1])]
            else:
                for j in range(cm.shape[1]):
                    cm_val_fmt[i, j] = find_val_fmt(cm[i, j])
        return cm_val_fmt

    def plot_confusion_matrix(self, cm, labels, name_plot='confusion_matrix.png', cmap="YlGn"):
        """ Plot a confusion matrix with the number of observation in the whole input dataset.

        Parameters
        ----------
        cm : np.array
            Confusion matrix.
        labels : list of str
            Labels for each class.
        name_plot : str, optional
            Name of the output file, by default 'confusion_matrix.png'
        cmap : str, optional
            colors to use in the plot, by default "YlGn"

        Returns
        -------
        str
            Ouput path of the image containing the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        cbarlabel = 'Coefficients values'

        im, _ = self.heatmap(cm, labels, labels, ax=ax, cmap=cmap, cbarlabel=cbarlabel)
        # Rewrite cm with strings in order to fit the values into the figure.

        cm_val_fmt = self.get_cm_val_fmt(cm)
        _ = self.annotate_heatmap(im, valfmt=cm_val_fmt)

        fig.tight_layout(pad=3)
        output_path = os.path.join(self.output_path, name_plot)
        plt.savefig(output_path)
        return output_path

    def plot_norm_and_value_cms(self, cm, labels, name_plot='norm_and_values_cms.png',
                                per_class_norm=True, cmap="YlGn"):
        """Plot a confusion matrix with the number of observation and also another one with values
        normalized (per class or by the whole cm).

        Parameters
        ----------
        cm : np.array
            Confusion matrix.
        labels : list of str
            Labels for each class.
        name_plot : str, optional
            Name of the output file, by default 'confusion_matrix.png'
        per_class_norm : bool, optional
            normalize per class or by the whole values in the cm, by default True
        cmap : str, optional
            colors to use in the plot, by default "YlGn"

        Returns
        -------
        str
            Ouput path of the image containing the plot.
        """

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
        cbarlabel = 'Coefficients values'

        # On ax0, normalize cm
        if not per_class_norm:
            a = cm.astype('float')
            b = np.sum(cm.flatten())
            cm_norm = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        else:
            a = cm.astype('float')
            b = cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

        im0, _ = self.heatmap(cm_norm, labels, labels, ax=axs[1], cmap=cmap, cbarlabel=cbarlabel)
        cm_val_fmt_norm = self.get_cm_val_fmt(cm_norm, mark_no_data=True)
        _ = self.annotate_heatmap(im0, data=np.round(cm_norm, decimals=3), valfmt=cm_val_fmt_norm)
        if not per_class_norm:
            axs[1].set_title('Normalized values', y=-0.1, pad=-14, fontsize=12)
        else:
            axs[1].set_title('Normalized per actual class values', y=-0.1, pad=-14, fontsize=12)

        im1, _ = self.heatmap(cm, labels, labels, ax=axs[0], cmap=cmap, cbarlabel=cbarlabel)
        cm_val_fmt = self.get_cm_val_fmt(cm)
        _ = self.annotate_heatmap(im1, valfmt=cm_val_fmt)
        axs[0].set_title('Number of observations', y=-0.1, pad=-14, fontsize=12)

        fig.tight_layout(pad=2)
        output_path = os.path.join(self.output_path, name_plot)
        plt.savefig(output_path)
        return output_path
