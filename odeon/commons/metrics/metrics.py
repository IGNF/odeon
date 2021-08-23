"""
Metrics tool to analyse the quality of a model's predictions.
Compute metrics, plot confusion matrices (cms) and ROC curves.
This tool handles binary, multi-classes and multi-labels cases.
The metrics computed are in each case :

* Binary case:
    - Accuracy
    - Precision
    - Recall
    - Specificity
    - F1 Score / Dice
    - IoU
    - AUC Score
    - Calibration Curve
    - KL Divergence

* Multi-class case:
    - Per class: same metrics as the binary case for each class.
    - Macro : same metrics as the binary case for the sum of all classes.
    - Micro : Precision, Recall, F1 Score and IoU.

* Multi-labels case:
    - Same as the multi-class case but with a threshold for each class insteas to use argmax to binarise predictions.
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
DEFAULTS_VARS = {'threshold': 0.5,
                 'threshold_range': np.arange(0.1, 1.1, 0.1),
                 'weights': None,
                 'nb_calibration_bins': 10,
                 'bit_depth': '8 bits',
                 'batch_size': 1,
                 'num_workers': 1,
                 'normalize': True,
                 'get_metrics_per_patch': True,
                 'get_ROC_PR_curves': True,
                 'get_calibration_curves': True,
                 'get_hists_per_metrics': True}


class Metrics(ABC):

    def __init__(self,
                 dataset,
                 output_path,
                 type_classifier,
                 output_type=None,
                 class_labels=None,
                 weights=DEFAULTS_VARS['weights'],
                 threshold=DEFAULTS_VARS['threshold'],
                 threshold_range=DEFAULTS_VARS['threshold_range'],
                 bit_depth=DEFAULTS_VARS['bit_depth'],
                 nb_calibration_bins=DEFAULTS_VARS['nb_calibration_bins'],
                 batch_size=DEFAULTS_VARS['batch_size'],
                 num_workers=DEFAULTS_VARS['num_workers'],
                 normalize=DEFAULTS_VARS['normalize'],
                 get_metrics_per_patch=DEFAULTS_VARS['get_metrics_per_patch'],
                 get_ROC_PR_curves=DEFAULTS_VARS['get_ROC_PR_curves'],
                 get_calibration_curves=DEFAULTS_VARS['get_calibration_curves'],
                 get_hists_per_metrics=DEFAULTS_VARS['get_hists_per_metrics']):

        if not os.path.exists(output_path):
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Output folder ${output_path} does not exist.")
        elif not os.path.isdir(output_path):
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Output path ${output_path} should be a folder.")
        else:
            self.output_path = output_path

        if output_type in ['md', 'json', 'html']:
            self.output_type = output_type
        else:
            LOGGER.error('ERROR: the output file can only be in md, json, html.')
            self.output_type = 'html'

        self.type_classifier = type_classifier.lower()
        self.dataset = dataset
        self.nbr_class = self.dataset.nbr_class

        if all(class_labels):
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
        self.threshold_range = threshold_range
        self.bit_depth = bit_depth
        self.nb_calibration_bins = nb_calibration_bins
        self.bins = np.linspace(0., 1. + 1e-8, self.nb_calibration_bins + 1)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize

        self.get_metrics_per_patch = get_metrics_per_patch
        self.get_ROC_PR_curves = get_ROC_PR_curves
        self.get_calibration_curves = get_calibration_curves
        self.get_hists_per_metrics = get_hists_per_metrics
        self.plot_ids = 0
        self.metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'IoU', 'FPR']

        self.depth_dict = {'keep':  1,
                           '8 bits': 255,
                           '12 bits': 4095,
                           '14 bits': 16383,
                           '16 bits': 65535}

        self.type_prob, self.in_prob_range = self.get_info_pred()

        if not self.get_ROC_PR_curves or self.type_prob == 'hard':
            self.threshold_range = [self.threshold]

        assert self.threshold in self.threshold_range, 'Threshold should be in the threshold range list.'

        if self.output_type == 'json':
            self.dict_export = {}
            self.dict_export['params'] = {'class_labels': self.class_labels,
                                          'threshold': self.threshold,
                                          'threshold_range': self.threshold_range
                                          if isinstance(self.threshold_range, list) else self.threshold_range.tolist(),
                                          'bins': self.bins.tolist()}

        self.report = Report_Factory(self)

    def __call__(self):
        self.report.create_report()

    @abstractmethod
    def create_data_for_metrics(self):
        pass

    @abstractmethod
    def get_metrics_from_cm(self):
        pass

    def binarize(self, type_classifier, prediction, mask=None, threshold=None):
        pred = prediction.copy()
        if not self.in_prob_range:
            pred = self.to_prob_range(pred)
        if type_classifier == 'multiclass':
            assert mask is not None
            return np.argmax(mask, axis=2), np.argmax(pred, axis=2)
        elif type_classifier == 'binary':
            if self.type_prob == 'soft':
                assert threshold is not None
                pred[pred < threshold] = 0
                pred[pred >= threshold] = 1
            return pred
        else:
            LOGGER.error('ERROR: type_classifier should be Binary or Multiclass')

    def get_confusion_matrix(self, truth, pred, nbr_class=None):
        """
        Return confusion matrix
        In binary case:
         [['tp' 'fn']
          ['fp' 'tn']]

        Parameters
        ----------
        truth : [type]
            [description]
        pred : [type]
            [description]

        Returns
        -------
        [type]
            [description]
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
        return np.flip(cm)

    def get_metrics_from_obs(self, tp, fn, fp, tn):

        # Accuracy
        if tp != 0.0 and tn != 0.0 and tp + tn != 0.0 and tp + fp + tn + fn != 0:
            accuracy = (tp + tn) / (tp + fp + tn + fn)
        else:
            accuracy = 0.0

        # Specificity
        if tn != 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0

        # Precision, Recall, F1-Score and IoU
        if tp != 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = (2 * tp) / (2 * tp + fp + fn)
            iou = tp / (tp + fp + fn)
        else:
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
            iou = 0.0

        if fp != 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0.0

        return {'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'F1-Score': f1_score,
                'IoU': iou,
                'FPR': fpr}

    def get_info_pred(self):
        """
            Tests on the first tenth of the predictions to check if inputs preds are in soft or in hard,
            and if pixels values are the expected range value.
        """
        nuniques = 0
        maxu = - float('inf')
        max_steps = len(self.dataset) // 10

        for i in range(max_steps):
            sample = self.dataset[i]
            pred = sample['pred']
            cr_nuniques = np.unique(pred.flatten())
            if len(cr_nuniques) > nuniques:
                nuniques = len(cr_nuniques)
            if max(cr_nuniques) > maxu:
                maxu = max(cr_nuniques)

        type_prob = 'soft' if nuniques > self.nbr_class else 'hard'
        in_prob_range = True if maxu <= 1 else False
        return type_prob, in_prob_range

    def to_prob_range(self, value):
        return value / self.depth_dict[self.bit_depth]

    def heatmap(self, data, row_labels, col_labels, ax=None,
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

    def annotate_heatmap(self, im, data=None, valfmt="{x:.3f}",
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

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    def plot_confusion_matrix(self, cm, labels, name_plot='confusion_matrix.png', cmap="YlGn"):

        if self.normalize:
            cm = cm.astype('float') / np.sum(cm.flatten())
        else:
            length_dict = {0: (1, ''),
                           3: (1000, 'k'),
                           6: (1000000, 'm'),
                           9: (1000000000, 'g'),
                           12: (1000000000000, 't')}
            max_length = len(str(int(min(cm.flatten()))))

            divider = 0
            unit_char = None

            for length in length_dict.keys():
                if max_length < length:
                    divider = length_dict[length][0]
                    unit_char = length_dict[length][1]
                    break

            cm = np.round(cm / divider, decimals=2)

        fig, ax = plt.subplots(figsize=(10, 6))
        cbarlabel = 'Coefficients values'
        im, _ = self.heatmap(cm, labels, labels, ax=ax, cmap=cmap, cbarlabel=cbarlabel)

        if self.normalize:
            _ = self.annotate_heatmap(im)
        else:
            valfmt = '{x:n}' + unit_char
            _ = self.annotate_heatmap(im, valfmt=valfmt)

        fig.tight_layout(pad=3)
        plt.title('Predicted class', fontsize=10)

        output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
        plt.savefig(output_path)
        return output_path
