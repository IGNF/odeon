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
import matplotlib.pyplot as plt
import itertools
from abc import ABC, abstractmethod
from odeon import LOGGER
from odeon.commons.reports.report_factory import Report_Factory

FIGSIZE = (8, 6)
DEFAULTS_VARS = {'threshold': 0.5,
                 'threshold_range': np.arange(0, 1.0025, 0.0025),
                 'nb_calibration_bins': 10,
                 'bit_depth': '8 bits',
                 'batch_size': 1,
                 'num_workers': 1}

from torch.utils.data import DataLoader

metrics_dataloader = DataLoader(self.dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
for sample in metrics_dataloader:


class Metrics(ABC):

    def __init__(self,
                 masks,
                 preds,
                 output_path,
                 type_classifier,
                 nbr_class,
                 class_labels=None,
                 threshold=DEFAULTS_VARS['threshold'],
                 threshold_range=DEFAULTS_VARS['threshold_range'],
                 bit_depth=DEFAULTS_VARS['bit_depth'],
                 nb_calibration_bins=DEFAULTS_VARS['nb_calibration_bins'],
                 batch_size=DEFAULTS_VARS['batch_size'],
                 num_workers=DEFAULTS_VARS['num_workers']):

        self.masks = masks
        self.preds = preds
        self.output_path = output_path
        self.type_classifier = type_classifier
        self.nbr_class = nbr_class

        if all(class_labels):
            self.class_labels = class_labels
        else:
            self.class_labels = [f'class {i + 1}' for i in range(self.nbr_class)]

        self.class_ids = np.arange(self.nbr_class)  # Each class is identified by a number
        self.threshold = threshold
        self.threshold_range = threshold_range
        self.bit_depth = bit_depth
        self.nb_calibration_bins = nb_calibration_bins
        self.bins = np.linspace(0., 1. + 1e-8, self.nb_calibration_bins + 1)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'IoU', 'FPR']

        self.depth_dict = {'keep':  1,
                           '8 bits': 255,
                           '12 bits': 4095,
                           '14 bits': 16383,
                           '16 bits': 65535}

        self.type_prob, self.in_prob_range = self.get_info_pred()
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
        if type_classifier == 'Multiclass':
            assert mask is not None
            return np.argmax(mask, axis=2), np.argmax(pred, axis=2)
        elif type_classifier == 'Binary':
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
        if tp != 0 or tn != 0:
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
        pred_samples = self.preds[:len(self.preds)//10]
        if len(pred_samples) == 1:
            pred_samples = list(pred_samples)
        nuniques = 0
        maxu = - float('inf')
        for pred in pred_samples:
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

    def plot_confusion_matrix(self,
                              cm,
                              nbr_class=None,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True,
                              name_plot='confusion_matrix.png'):
        """
        Given a confusion matrix (cm) in np.ndarray, return its plot.

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
        normalize:    If False, plot the raw numbers
                      If True, plot the proportions
        """
        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=FIGSIZE)
        plt.imshow(cm, cmap=cmap)
        plt.title(title)
        plt.colorbar()
        if nbr_class is not None:
            tick_marks = np.arange(nbr_class)
            plt.xticks(tick_marks, labels=[0, 1])
            plt.yticks(tick_marks, labels=[0, 1])
        else:
            tick_marks = np.arange(self.nbr_class)
            plt.xticks(tick_marks, labels=self.class_labels)
            plt.yticks(tick_marks, self.class_labels)

        if normalize:
            cm = cm.astype('float') / np.sum(cm.flatten())

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.3f}".format(cm[i, j]), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh and cmap == plt.get_cmap('Blues') else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout(pad=3)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
        plt.savefig(output_path)
        return output_path
