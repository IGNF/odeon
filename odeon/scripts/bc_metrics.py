import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from metrics import Metrics, DEFAULTS_VARS
from tqdm import tqdm

FIGSIZE = (8, 6)


class BC_Metrics(Metrics):

    def __init__(self,
                 masks,
                 preds,
                 output_path,
                 nbr_class,
                 class_labels=None,
                 threshold=DEFAULTS_VARS['threshold'],
                 threshold_range=DEFAULTS_VARS['threshold_range'],
                 bit_depth=DEFAULTS_VARS['bit_depth'],
                 nb_calibration_bins=DEFAULTS_VARS['nb_calibration_bins']):

        super().__init__(masks=masks,
                         preds=preds,
                         output_path=output_path,
                         nbr_class=nbr_class,
                         class_labels=class_labels,
                         threshold=threshold,
                         threshold_range=threshold_range,
                         bit_depth=bit_depth,
                         nb_calibration_bins=nb_calibration_bins)

        self.df_thresholds, self.cms, self.df_report_metrics = self.create_data_for_metrics()
        self.get_metrics_by_threshold()

    def create_data_for_metrics(self):
        df_thresholds = pd.DataFrame(index=range(len(self.threshold_range)),
                                     columns=(['threshold'] + self.metrics_names))
        df_thresholds['threshold'] = self.threshold_range
        cms = {}
        df_report_metrics = pd.DataFrame(index=['Values'], columns=self.metrics_names[:-1])
        return df_thresholds, cms, df_report_metrics

    def binarize(self, prediction, threshold):
        pred = prediction.copy()
        if not self.in_prob_range:
            pred = self.to_prob_range(pred)

        pred[pred < threshold] = 0
        pred[pred >= threshold] = 1

        return pred

    def get_metrics_by_threshold(self):

        for threshold in tqdm(self.threshold_range, leave=True):
            self.cms[threshold] = np.zeros([self.nbr_class, self.nbr_class])
            for mask, pred in zip(self.masks, self.preds):
                pred = self.binarize(pred, threshold)
                cm = self.get_confusion_matrix(mask.flatten(), pred.flatten())
                self.cms[threshold] += cm
            cr_metrics = self.get_metrics_from_cm(self.cms[threshold])

            for metric, metric_value in cr_metrics.items():
                self.df_thresholds.loc[self.df_thresholds['threshold'] == threshold, metric] = metric_value

        self.df_report_metrics.loc['Values'] = \
            self.df_thresholds.loc[self.df_thresholds['threshold'] == self.threshold, self.metrics_names[:-1]].values

    def get_metrics_from_cm(self, cm):
        tp, fn, fp, tn = cm.ravel()
        return self.get_metrics_from_obs(tp, fn, fp, tn)

    def plot_PR_curve(self, precision, recall, name_plot='pr_curve.png'):

        precision = np.array([1 if p == 0 and r == 0 else p for p, r in zip(precision, recall)])
        idx = np.argsort(recall)
        recall, precision = recall[idx], precision[idx]
        pr_auc = auc(recall, precision)

        plt.figure(figsize=FIGSIZE)
        plt.title('Precision-Recall Curve')
        plt.plot(recall, precision, label='AUC = %0.3f' % pr_auc)
        plt.plot([1, 0], [0, 1], 'r--')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.legend()
        plt.grid(True)
        output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
        plt.savefig(output_path)
        return output_path

    def plot_ROC_curve(self, fpr, tpr, name_plot='roc_curve.png'):

        # Sorted fpr in increasing order to plot it as the abscisses values of the curve.
        # fpr, tpr = np.insert(fpr.to_numpy(), 0, 0), np.insert(tpr.to_numpy(), 0, 0)
        fpr, tpr = fpr[::-1], tpr[::-1]
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=FIGSIZE)
        plt.title('Roc Curve')
        plt.plot(fpr, tpr, label='AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()
        plt.grid(True)
        output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
        plt.savefig(output_path)
        return output_path

    def plot_calibration_curve(self, n_bins=None, name_plot='calibration_curves.png'):

        if n_bins is None:
            n_bins = self.nb_calibration_bins

        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
        bins_labels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        hist_counts = np.zeros(len(bins) - 1)
        bin_sums = np.zeros(len(bins))
        bin_true = np.zeros(len(bins))
        bin_total = np.zeros(len(bins))

        for mask, pred in tqdm(zip(self.masks, self.preds), desc='Calibration curves', leave=True):
            pred.copy()
            if not self.in_prob_range:
                pred = self.to_prob_range(pred)
            hist_counts += np.histogram(pred.flatten(), bins=bins)[0]
            binids = np.digitize(pred.flatten(), bins) - 1
            bin_sums += np.bincount(binids, weights=pred.flatten(), minlength=len(bins))
            bin_true += np.bincount(binids, weights=mask.flatten(), minlength=len(bins))
            bin_total += np.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]

        plt.figure(figsize=(16, 8))
        plt.subplot(211)
        plt.plot(prob_true, prob_pred, "s-")
        plt.subplot(212)
        plt.bar(list(range(len(hist_counts))), hist_counts, tick_label=bins_labels)

        output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
        plt.savefig(output_path)
        return output_path
