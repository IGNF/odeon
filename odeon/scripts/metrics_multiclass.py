import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from metrics import Metrics, DEFAULTS_VARS
# from tqdm import tqdm
# from odeon.commons.metrics import Metrics


class Metrics_Multiclass(Metrics):

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
                 nb_calibration_bins=DEFAULTS_VARS['nb_calibration_bins']):

        super().__init__(masks=masks,
                         preds=preds,
                         output_path=output_path,
                         type_classifier=type_classifier,
                         nbr_class=nbr_class,
                         class_labels=class_labels,
                         threshold=threshold,
                         threshold_range=threshold_range,
                         bit_depth=bit_depth,
                         nb_calibration_bins=nb_calibration_bins)

        self.df_report_classes, self.df_report_micro, self.df_report_macro = self.create_data_for_metrics()
        self.cm_micro = self.get_cm_micro()
        self.metrics_by_class, self.metrics_micro, self.metrics_macro = self.get_metrics_from_cm()
        self.metrics_to_df_reports()

    def create_data_for_metrics(self):
        df_report_classes = pd.DataFrame(index=[class_name for class_name in self.class_labels],
                                         columns=self.metrics_names[:-1])
        df_report_micro = pd.DataFrame(index=['Values'],
                                       columns=['Precision', 'Recall', 'F1-Score', 'IoU'])
        df_report_macro = pd.DataFrame(index=['Values'],
                                       columns=self.metrics_names[:-1])
        return df_report_classes, df_report_micro, df_report_macro

    def get_cm_micro(self):
        cm_micro = np.zeros([self.nbr_class, self.nbr_class])
        for mask, pred in zip(self.masks, self.preds):
            mask, pred = self.binarize(self.type_classifier, pred, mask=mask)
            cm = self.get_confusion_matrix(mask.flatten(), pred.flatten())
            cm_micro += cm
        return cm_micro

    def get_obs_by_class_from_cm(self, cm):
        obs_by_class = {}
        for i, class_i in enumerate(self.class_labels):
            obs_by_class[class_i] = {'tp': cm[i, i],
                                     'fn': np.sum(cm[i, :]) - cm[i, i],
                                     'fp': np.sum(cm[:, i]) - cm[i, i],
                                     'tn': np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i])}
        return obs_by_class

    def get_obs_micro_from_cm(self, cm):
        obs_micro = {}
        obs_micro = {'tp': np.sum(np.diag(cm)),
                     'fn': np.sum(np.triu(cm, k=1)),
                     'fp': np.sum(np.tril(cm, k=-1))}
        return obs_micro

    def get_metrics_from_cm(self):
        obs_by_class = self.get_obs_by_class_from_cm(self.cm_micro)
        obs_micro = self.get_obs_micro_from_cm(self.cm_micro)

        self.cms_classes = np.zeros([self.nbr_class, 2, 2])

        metrics_by_class = {}
        for i, class_i in enumerate(self.class_labels):
            self.cms_classes[i] = np.array([[obs_by_class[class_i]['tp'], obs_by_class[class_i]['fn']],
                                            [obs_by_class[class_i]['fp'], obs_by_class[class_i]['tn']]])
            metrics_by_class[class_i] = self.get_metrics_from_obs(obs_by_class[class_i]['tp'],
                                                                  obs_by_class[class_i]['fn'],
                                                                  obs_by_class[class_i]['fp'],
                                                                  obs_by_class[class_i]['tn'])
        self.cm_macro = np.sum(self.cms_classes, axis=0)

        # We will only look at Precision, Recall, F1-Score and IoU.
        # Others metrics will be false because we don't have any TN in micro.
        metrics_micro = self.get_metrics_from_obs(obs_micro['tp'],
                                                  obs_micro['fn'],
                                                  obs_micro['fp'],
                                                  0)

        metrics_macro = self.get_metrics_from_obs(self.cm_macro[0][0],
                                                  self.cm_macro[0][1],
                                                  self.cm_macro[1][0],
                                                  self.cm_macro[1][1])

        return metrics_by_class, metrics_micro, metrics_macro

    def metrics_to_df_reports(self):
        for class_i in self.class_labels:
            self.df_report_classes.loc[class_i] = list(self.metrics_by_class[class_i].values())[:-1]

        self.df_report_micro.loc['Values'] = [self.metrics_micro['Precision'],
                                              self.metrics_micro['Recall'],
                                              self.metrics_micro['F1-Score'],
                                              self.metrics_micro['IoU']]

        self.df_report_macro.loc['Values'] = list(self.metrics_macro.values())[:-1]

    def plot_ROC_PR_per_class(self, name_plot='roc_pr_curves.png'):

        vect_classes = {}
        for i, class_i in enumerate(self.class_labels):
            vects = {'Recall': [],
                     'FPR': [],
                     'Precision': []}
            for threshold in self.threshold_range:
                cms = np.zeros([2, 2])
                for mask, pred in zip(self.masks, self.preds):
                    class_mask = mask[:, :, i]
                    class_pred = pred[:, :, i]
                    bin_pred = self.binarize('Binary', class_pred, threshold=threshold)
                    cm = self.get_confusion_matrix(class_mask.flatten(), bin_pred.flatten(), nbr_class=2)
                    cms += cm
                cr_metrics = self.get_metrics_from_obs(cms[0][0],
                                                       cms[0][1],
                                                       cms[1][0],
                                                       cms[1][1])
                vects['Recall'].append(cr_metrics['Recall'])
                vects['FPR'].append(cr_metrics['FPR'])
                vects['Precision'].append(cr_metrics['Precision'])
            vect_classes[class_i] = vects

        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        for class_i in self.class_labels:
            fpr = np.array(vect_classes[class_i]['FPR'])[::-1]
            tpr = np.array(vect_classes[class_i]['Recall'])[::-1]
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_i} AUC = {round(roc_auc, 3)}')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('Roc Curves')
        plt.legend()
        plt.grid(True)

        plt.subplot(122)
        for class_i in self.class_labels:
            precision = np.array(vect_classes[class_i]['Precision'])
            recall = np.array(vect_classes[class_i]['Recall'])
            precision = np.array([1 if p == 0 and r == 0 else p for p, r in zip(precision, recall)])
            idx = np.argsort(recall)
            recall, precision = recall[idx], precision[idx]
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{class_i} AUC = {round(pr_auc, 3)}')
        plt.plot([1, 0], [0, 1], 'r--')
        plt.title('Precision-Recall Curve')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
        plt.savefig(output_path)
        return output_path

    def plot_calibration_curve(self, n_bins=None, name_plot='multiclass_calibration_curves.png'):

        if n_bins is None:
            n_bins = self.nb_calibration_bins
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
        dict_hist_counts, dict_prob_true, dict_prob_pred = {}, {}, {}

        for i, class_i in enumerate(self.class_labels):
            hist_counts = np.zeros(len(bins) - 1)
            bin_sums = np.zeros(len(bins))
            bin_true = np.zeros(len(bins))
            bin_total = np.zeros(len(bins))

            for mask, pred in zip(self.masks, self.preds):
                pred = pred.copy()[:, :, i]
                mask = mask.copy()[:, :, i]

                if not self.in_prob_range:
                    pred = self.to_prob_range(pred)
                # For plot 2, bincounts for histogram of prediction
                hist_counts += np.histogram(pred.flatten(), bins=bins)[0]

                # For plot 1
                # Indices of the bins where the predictions will be in there.
                binids = np.digitize(pred.flatten(), bins) - 1
                # Bins counts of indices times the values of the predictions.
                bin_sums += np.bincount(binids, weights=pred.flatten(), minlength=len(bins))
                # Bins counts of indices times the values of the masks.
                bin_true += np.bincount(binids, weights=mask.flatten(), minlength=len(bins))
                # Total number observation per bins.
                bin_total += np.bincount(binids, minlength=len(bins))

            nonzero = bin_total != 0  # Avoid to display null bins.
            prob_true = bin_true[nonzero] / bin_total[nonzero]
            prob_pred = bin_sums[nonzero] / bin_total[nonzero]

            dict_hist_counts[class_i] = hist_counts
            dict_prob_true[class_i] = prob_true
            dict_prob_pred[class_i] = prob_pred

        # Normalize dict_hist_counts to put the values between 0 and 1:
        total_pixel = np.sum(dict_hist_counts[self.class_labels[0]])
        dict_hist_counts = {key: value / total_pixel for key, value in dict_hist_counts.items()}

        plt.figure(figsize=(16, 8))
        # Plot 1: calibration curves
        plt.subplot(211)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for class_i in self.class_labels:
            plt.plot(dict_prob_true[class_i], dict_prob_pred[class_i], "s-", label=class_i)
        plt.legend(loc="lower right")
        plt.title('Calibration plots  (reliability curve)')
        plt.ylabel('Fraction of positives')
        plt.xlabel('Probalities')

        # Plot 2: Hist of predictions distributions
        plt.subplot(212)
        for class_i in self.class_labels:
            plt.hist(dict_hist_counts[class_i], bins=bins, histtype="step", label=class_i, lw=2)
        plt.ylabel('Count')
        plt.xlabel('Mean predicted value')
        plt.legend(loc="upper center")
        plt.tight_layout(pad=3)

        output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
        plt.savefig(output_path)
        return output_path
