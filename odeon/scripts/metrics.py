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
    - F1 Score
    - IoU
    - Dice
    - AUC Score
    - Expected Calibration Error (ECE)
    - Calibration Curve
    - KL Divergence

* Multi-class case:
    - 1 versus all: same metrics as the binary case for each class.
    - Macro (1 versus all then all cms stacked): same metrics as the binary case for the sum of all classes.
    - Micro : Precision, Recall, F1 Score (confusion matrix but no ROC curve).

* Multi-labels case:
    - Same as the multi-class case but without the global confusion matrix in  micro analysis.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.calibration import calibration_curve
import itertools
from abc import ABC, abstractmethod
# from odeon.commons.metrics import Metrics

DEFAULTS_VARS = {'roc_range': np.arange(0, 1.1, 0.1),
                 'nb_calibration_bins': 10,
                 'bit_depth': '8 bits',
                 'batch_size': 1,
                 'num_workers': 1}


class Metrics(ABC):

    def __init__(self,
                 mask_files,
                 pred_files,
                 output_path,
                 bit_depth=DEFAULTS_VARS['bit_depth'],
                 nb_calibration_bins=DEFAULTS_VARS['nb_calibration_bins'],
                 batch_size=DEFAULTS_VARS['batch_size'],
                 num_workers=DEFAULTS_VARS['num_workers']):

        self.mask_files = mask_files
        self.pred_files = pred_files
        self.output_path = output_path
        self.bit_depth = bit_depth
        self.nb_calibration_bins = nb_calibration_bins
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.depth_dict = {'keep':  1,
                           '8 bits': 255,
                           '12 bits': 4095,
                           '14 bits': 16383,
                           '16 bits': 65535}

    def __call__(self):
        pass

    @abstractmethod
    def create_data_for_metrics(self):
        pass

    @abstractmethod
    def binarize(self):
        pass

    @abstractmethod
    def get_metrics_from_cm(self):
        pass

    def plot_confusion_matrix(self,
                              cm,
                              target_names=None,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True):
        """
        from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels.
        Given a sklearn confusion matrix (cm), make a nice plot.

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                    the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                    see http://matplotlib.org/examples/color/colormaps_reference.html
                    plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                    If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                # sklearn.metrics.confusion_matrix
                            normalize    = True,                # show proportions
                            target_names = y_labels_vals,       # list of names of the classes
                            title        = best_estimator_name) # title of graph
        """
        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()

    def plot_PR_curve(self, precision, recall, generate=True, name_plot='pr_curve.png'):
        plt.figure(figsize=(7, 5))
        plt.title('Precision-Recall Curve')
        plt.plot(recall, precision)
        plt.plot([1, 0], [0, 1], 'r--')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.legend()

        if generate:
            output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
            plt.savefig(output_path)
            return output_path
        else:
            plt.show()

    def plot_ROC_curve(self, fpr, tpr, generate=True, name_plot='roc_curve.png'):

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))
        plt.title('Roc Curve')
        plt.plot(fpr, tpr, label='AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()

        if generate:
            output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
            plt.savefig(output_path)
            return output_path
        else:
            plt.show()

    def plot_calibration_curve(self, mask, pred, n_bins=None):
        if n_bins is None:
            n_bins = self.nb_calibration_bins
        fraction_of_positives, mean_predicted_value = calibration_curve(mask, pred, n_bins)
        pass
