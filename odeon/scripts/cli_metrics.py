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
import csv
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import itertools
from odeon import LOGGER
from odeon.commons.core import BaseTool
from odeon.commons.image import image_to_ndarray
from odeon.commons.exception import OdeonError, ErrorCodes
from abc import ABC, abstractmethod
# from odeon.commons.metrics import Metrics

ROC_RANGE = np.arange(0, 1.1, 0.1)
BATCH_SIZE = 1
NUM_WORKERS = 1


def plot_confusion_matrix(cm,
                          target_names,
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
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def Metrics_Factory(type_classifier):
    """Function assigning which class should be used.

    Parameters
    ----------
    input_object : Statistics/Metrics
        Input object to use to create a report.

    Returns
    -------
    Stats_Report/Metric_Report
        An object making the report.
    """
    metrics = {"Binary case": BC_Metrics,
               "Multi-class mono-label": MC_1L_Metrics,
               "Multi-class multi-label": MC_ML_Metrics}

    return metrics[type_classifier]


class Metrics(ABC):

    def __init__(self,
                 mask_files,
                 pred_files,
                 output_path,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):

        self.mask_files = mask_files
        self.pred_files = pred_files
        self.output_path = output_path
        self.batch_size = batch_size
        self.num_workers = num_workers

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


class BC_Metrics(Metrics):

    def __init__(self,
                 mask_files,
                 pred_files,
                 output_path,
                 threshold,
                 roc_range=ROC_RANGE,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):

        super().__init__(mask_files=mask_files,
                         pred_files=pred_files,
                         output_path=output_path,
                         batch_size=batch_size,
                         num_workers=num_workers)
        self.nbr_class = 2  # Crée un moyen de récupérer proprement le nombre de classes.
        self.cms = np.zeros((self.nbr_class, self.nbr_class))
        self.threshold = threshold
        self.roc_range = roc_range
        self.df_dataset = self.create_data_for_metrics()
        self.get_metrics()
        print(self.df_dataset.head())

    def create_data_for_metrics(self):
        header = ['pred', 'mask', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'IoU']
        df_dataset = pd.DataFrame(index=range(len(self.pred_files)), columns=header)
        df_dataset['pred'] = self.pred_files
        df_dataset['mask'] = self.mask_files
        return df_dataset

    def update_df_dataset(self, pred_file, metrics):
        self.df_dataset.loc[self.df_dataset['pred'] == pred_file, ['Accuracy']] = metrics['Accuracy']
        self.df_dataset.loc[self.df_dataset['pred'] == pred_file, ['Precision']] = metrics['Precision']
        self.df_dataset.loc[self.df_dataset['pred'] == pred_file, ['Recall']] = metrics['Recall']
        self.df_dataset.loc[self.df_dataset['pred'] == pred_file, ['Specificity']] = metrics['Specificity']
        self.df_dataset.loc[self.df_dataset['pred'] == pred_file, ['F1-Score']] = metrics['F1-Score']
        self.df_dataset.loc[self.df_dataset['pred'] == pred_file, ['IoU']] = metrics['IoU']

    def binarize(self, prediction, threshold):
        tmp = prediction.copy()
        tmp[prediction > threshold] = 1
        tmp[prediction <= threshold] = 0
        return tmp.copy()

    def get_metrics(self):
        for mask_file, pred_file in zip(self.mask_files, self.pred_files):
            mask = image_to_ndarray(mask_file)
            pred = image_to_ndarray(pred_file)

            pred = self.binarize(pred, self.threshold)
            cm = confusion_matrix(mask.flatten(), pred.flatten())
            self.cms += cm
            cr_metrics = self.get_metrics_from_cm(cm)
            self.update_df_dataset(pred_file, cr_metrics)

    def get_metrics_from_cm(self, cm):
        tn, fp, fn, tp = cm.ravel()

        # Accuracy
        if tp != 0 or tn != 0:
            accuracy = (tp + tn) / (tp + fp + tn + fn)
        else:
            accuracy = 0

        # Specificity
        if tn != 0:
            specificity = tn/(tn + fp)
        else:
            specificity = 0

        # Precision, Recall, F1-Score and IoU
        if tp != 0:
            precision = tp / (tp + fp)
            recall = tp/(tp + fn)
            f1_score = (2 * tp) / (2 * tp + fp + fn)
            iou = tp / (tp + fp + fn)
        else:
            precision = 0
            recall = 0
            f1_score = 0
            iou = 0

        return {'accuracy ': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'F1': f1_score,
                'IoU': iou}


class MC_1L_Metrics(Metrics):
    pass


class MC_ML_Metrics(Metrics):
    pass


class CLI_Metrics(BaseTool):

    def __init__(self,
                 mask_path,
                 pred_path,
                 output_path,
                 type_classifier,
                 threshold,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):

        self.mask_path = mask_path
        self.pred_path = pred_path
        self.output_path = output_path
        self.type_classifier = type_classifier
        self.threshold = threshold
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.mask_files, self.pred_files = self.files_from_input_paths()

        self.metrics = Metrics_Factory(self.type_classifier)(mask_files=self.mask_files,
                                                             pred_files=self.pred_files,
                                                             output_path=self.output_path,
                                                             threshold=self.threshold,
                                                             batch_size=self.batch_size,
                                                             num_workers=self.num_workers)

    def __call__(self):
        self.metrics()

    def files_from_input_paths(self):
        if not os.path.exists(self.mask_path):
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"Masks folder ${self.mask_path} does not exist.")
        elif not os.path.exists(self.pred_path):
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"Predictions folder ${self.pred_path} does not exist.")
        else:
            if os.path.isdir(self.mask_path) and os.path.isdir(self.pred_path):
                mask_files, pred_files = self.list_files_from_dir()
            else:
                LOGGER.error('ERROR: the input paths shoud point to dataset directories.')
        return mask_files, pred_files

    def read_csv_sample_file(self):
        mask_files = []
        pred_files = []

        with open(self.input_path) as csvfile:
            sample_reader = csv.reader(csvfile)
            for item in sample_reader:
                mask_files.append(item['msk_file'])
                pred_files.append(item['img_output_file'])
        return mask_files, pred_files

    def list_files_from_dir(self):
        mask_files, pred_files = [], []

        for msk, pred in zip(sorted(os.listdir(self.mask_path)), sorted(os.listdir(self.pred_path))):
            if msk == pred:
                mask_files.append(os.path.join(self.mask_path, msk))
                pred_files.append(os.path.join(self.pred_path, pred))
            else:
                LOGGER.warning(f'Problem of matching names between mask {msk} and prediction {pred}.')
        return mask_files, pred_files


if __name__ == '__main__':
    img_path = '/home/SPeillet/OCSGE/data/metrics/subset_binaire/img'
    mask_path = '/home/SPeillet/OCSGE/data/metrics/subset_binaire/msk'
    pred_path = '/home/SPeillet/OCSGE/data/metrics/subset_binaire/pred'
    output_path = '/home/SPeillet/OCSGE/data/metrics/subset_binaire/outputs'
    print('------------------------------------------------------------------------------ ')
    metrics = CLI_Metrics(mask_path, pred_path, output_path, type_classifier='Binary case', threshold=0.5)
    metrics()


