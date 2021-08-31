"""
Class to manage metrics in the binary case.
Will compute: metrics (Precision, Recall, Specifity, F1-Score, IoU), confusion matrix (cm),
ROC/PR curves, calibration curve and metrics histograms.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from odeon.commons.metrics.metrics import Metrics, DEFAULTS_VARS
from tqdm import tqdm

FIGSIZE = (8, 6)


class Metrics_Binary(Metrics):

    def __init__(self,
                 dataset,
                 output_path,
                 type_classifier,
                 class_labels=None,
                 output_type=DEFAULTS_VARS['output_type'],
                 weights=DEFAULTS_VARS['weights'],
                 threshold=DEFAULTS_VARS['threshold'],
                 threshold_range=DEFAULTS_VARS['threshold_range'],
                 bit_depth=DEFAULTS_VARS['bit_depth'],
                 nb_calibration_bins=DEFAULTS_VARS['nb_calibration_bins'],
                 get_normalize=DEFAULTS_VARS['get_normalize'],
                 get_metrics_per_patch=DEFAULTS_VARS['get_metrics_per_patch'],
                 get_ROC_PR_curves=DEFAULTS_VARS['get_ROC_PR_curves'],
                 get_calibration_curves=DEFAULTS_VARS['get_calibration_curves'],
                 get_hists_per_metrics=DEFAULTS_VARS['get_hists_per_metrics']):
        """
        Init function.
        Initialize the class attributes and create the dataframes to store the metrics.
        Scan the dataset, for every threshold given the threshold range (for ROC and PR curves), for each
        sample to compute confusion matrices (cms) and metrics.
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
            Here the classfier type should be 'binary'.
        output_type : str, optional
            Desired format for the output file. Could be json, md or html.
            A report will be created if the output type is html or md.
            If the output type is json, all the data will be exported in a dict in order
            to be easily reusable, by default html.
        class_labels : list of str, optional
            Label for each class in the dataset.
            If None the labels of the classes will be of type:  0 and 1 by default None
        weights : list of number, optional
            List of weights to balance the metrics.
            In the binary case the weights are not used in the metrics computation, by default None.
        threshold : float, optional
            Value between 0 and 1 that will be used as threshold to binarize data if they are soft.
            Use for macro, micro cms and metrics for all strategies, by default 0.5.
        threshold_range : list of float, optional
            List of values that will be used as a threshold when calculating the ROC and PR curves,
            by default np.arange(0.1, 1.1, 0.1).
        bit_depth : str, optional
            The number of bits used to represent each pixel in a mask/prediction, by default '8 bits'
        nb_calibration_bins : int, optional
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

        super().__init__(dataset,
                         output_path=output_path,
                         type_classifier=type_classifier,
                         class_labels=class_labels,
                         output_type=output_type,
                         weights=weights,
                         threshold=threshold,
                         threshold_range=threshold_range,
                         bit_depth=bit_depth,
                         nb_calibration_bins=nb_calibration_bins,
                         get_normalize=get_normalize,
                         get_metrics_per_patch=get_metrics_per_patch,
                         get_ROC_PR_curves=get_ROC_PR_curves,
                         get_calibration_curves=get_calibration_curves,
                         get_hists_per_metrics=get_hists_per_metrics)

        self.df_thresholds, self.cms, self.df_report_metrics = self.create_data_for_metrics()

        if self.get_metrics_per_patch:
            self.df_dataset = pd.DataFrame(index=range(len(self.dataset)),
                                           columns=(['name_file'] + self.metrics_names[:-1]))

        self.get_metrics_by_threshold()

        if get_metrics_per_patch:
            self.export_metrics_per_patch_csv()

    def create_data_for_metrics(self):
        """
        Create dataframes to store metrics for each strategy.

        Returns
        -------
        Tuple of pd.DataFrame
            Dataframes to store metrics for each strategy.
        """
        df_thresholds = pd.DataFrame(index=range(len(self.threshold_range)),
                                     columns=(['threshold'] + self.metrics_names))
        df_thresholds['threshold'] = self.threshold_range
        cms = {}
        df_report_metrics = pd.DataFrame(index=['Values'], columns=self.metrics_names[:-1])

        return df_thresholds, cms, df_report_metrics

    def get_metrics_by_threshold(self):
        """
        Function allowing to make a pass on the dataset in order to obtain confusion
        matrices by sample according to given thresholds. For each threshold, then the cms
        are added together to make only one cm by threshold.
        """
        hist_counts = np.zeros(len(self.bins) - 1)
        bin_sums = np.zeros(len(self.bins))
        bin_true = np.zeros(len(self.bins))
        bin_total = np.zeros(len(self.bins))

        for threshold in tqdm(self.threshold_range, desc='Tresholds', leave=False):
            self.cms[threshold] = np.zeros([self.nbr_class, self.nbr_class])

            dataset_index = 0
            for sample in self.dataset:
                mask, pred, name_file = sample['mask'], sample['pred'], sample['name_file']

                # Compute cm on every sample
                pred_cm = pred.copy()
                pred_cm = self.binarize(self.type_classifier, pred_cm, threshold=threshold)
                cm = self.get_confusion_matrix(mask.flatten(), pred_cm.flatten())
                self.cms[threshold] += cm

                # Compute metrics per patch and return an histogram of the values
                if self.get_metrics_per_patch and threshold == self.threshold:
                    sample_metrics = self.get_metrics_from_cm(cm)
                    self.df_dataset.loc[dataset_index, 'name_file'] = name_file
                    for name_column in self.metrics_names[:-1]:
                        self.df_dataset.loc[dataset_index, name_column] = sample_metrics[name_column]
                    dataset_index += 1

                # To calcultate info for calibrations curves only once.
                if threshold == self.threshold_range[0]:
                    pred_hist = pred.copy()
                    if not self.in_prob_range:
                        pred_hist = self.to_prob_range(pred_hist)
                    # bincounts for histogram of prediction
                    hist_counts += np.histogram(pred_hist.flatten(), bins=self.bins)[0]

                    # Indices of the bins where the predictions will be in there.
                    binids = np.digitize(pred_hist.flatten(), self.bins) - 1
                    # Bins counts of indices times the values of the predictions.
                    bin_sums += np.bincount(binids, weights=pred_hist.flatten(), minlength=len(self.bins))
                    # Bins counts of indices times the values of the masks.
                    bin_true += np.bincount(binids, weights=mask.flatten(), minlength=len(self.bins))
                    # Total number observation per bins.
                    bin_total += np.bincount(binids, minlength=len(self.bins))

            cr_metrics = self.get_metrics_from_cm(self.cms[threshold])

            for metric, metric_value in cr_metrics.items():
                self.df_thresholds.loc[self.df_thresholds['threshold'] == threshold, metric] = metric_value

        # Normalize hist_counts to put the values between 0 and 1:
        self.hist_counts = hist_counts / np.sum(hist_counts)
        nonzero = bin_total != 0  # Avoid to display null bins.
        self.prob_true = bin_true[nonzero] / bin_total[nonzero]
        self.prob_pred = bin_sums[nonzero] / bin_total[nonzero]

        self.df_report_metrics.loc['Values'] = \
            self.df_thresholds.loc[self.df_thresholds['threshold'] == self.threshold, self.metrics_names[:-1]].values

    def get_metrics_from_cm(self, cm):
        """
        Extract the metrics from a cm.

        Parameters
        ----------
        cm : np.array[type]
            Confusion matrix in micro strategy.

        Returns
        -------
        dict
            Dict with the metrics.
        """
        tp, fn, fp, tn = cm.ravel()
        return self.get_metrics_from_obs(tp, fn, fp, tn)

    def plot_ROC_PR_curves(self, name_plot='binary_roc_pr_curve.png'):
        """
        Plot (html/md output type) or export (json type) data on ROC and PR curves.
        For ROC curve, points (0, 0) and (1, 1) are added to data in order to have a curve
        which begin at the origin and finish in the top right of the image.
        Same thing for PR curve but with the points (0, 1) and (1, 0).

        Parameters
        ----------
        name_plot : str, optional
            Desired name to give to the output image, by default 'multiclass_roc_pr_curves.png'

        Returns
        -------
        str
            Output path where an image with the plot will be created.
        """
        fpr, tpr = self.df_thresholds['FPR'], self.df_thresholds['Recall']
        recall, precision = self.df_thresholds['Recall'], self.df_thresholds['Precision']

        # Sorted fpr in increasing order to plot it as the abscisses values of the curve.
        fpr, tpr = np.insert(fpr.to_numpy(), 0, 1), np.insert(tpr.to_numpy(), 0, 1)
        fpr, tpr = np.append(fpr, 0), np.append(tpr, 0)
        fpr, tpr = fpr[::-1], tpr[::-1]
        roc_auc = auc(fpr, tpr)

        precision = np.array([1 if p == 0 and r == 0 else p for p, r in zip(precision, recall)])
        idx = np.argsort(recall)
        recall, precision = recall[idx], precision[idx]
        recall = np.insert(recall.to_numpy(), 0, 0)
        recall = np.append(recall, 1)
        precision = np.insert(precision, 0, 1)
        precision = np.append(precision, 0)
        pr_auc = auc(recall, precision)

        if self.output_type == 'json':
            # Export ROC curve informations
            roc_dict = {}
            roc_dict['fpr'] = fpr.tolist()
            roc_dict['tpr'] = tpr.tolist()
            roc_dict['auc'] = roc_auc
            self.dict_export['ROC curve'] = roc_dict
            # Export PR curve informations
            pr_dict = {}
            pr_dict['precision'] = precision.tolist()
            pr_dict['recall'] = recall.tolist()
            pr_dict['auc'] = pr_auc
            self.dict_export['PR curve'] = pr_dict

        else:
            plt.figure(figsize=(16, 8))
            plt.subplot(121)
            plt.title('Roc Curve')
            plt.plot(fpr, tpr, label='AUC = %0.3f' % roc_auc)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend()
            plt.grid(True)

            plt.subplot(122)
            plt.title('Precision-Recall Curve')
            plt.plot(recall, precision, label='AUC = %0.3f' % pr_auc)
            plt.plot([1, 0], [0, 1], 'r--')
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.legend()
            plt.grid(True)

            output_path = os.path.join(self.output_path, name_plot)
            plt.savefig(output_path)
            return output_path

    def plot_calibration_curve(self, name_plot='binary_calibration_curves.png'):
        """
        Plot or export data on calibration curves.
        Plot for output type html and md, export the data in a dict for json.

        Parameters
        ----------
        name_plot : str, optional
            Name to give to the output plot, by default 'multiclass_calibration_curves.png'

        Returns
        -------
        str
            Output path where an image with the plot will be created.
        """
        if self.output_type == 'json':
            self.dict_export['calibration curve'] = {'prob_true': self.prob_true.tolist(),
                                                     'prob_pred': self.prob_pred.tolist(),
                                                     'hist_counts': self.hist_counts.tolist()}
        else:
            plt.figure(figsize=(16, 8))
            plt.subplot(211)
            # Plot 1: calibration curves
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.plot(self.prob_true, self.prob_pred, "s-", label="Class 1")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('Calibration plots  (reliability curve)')
            plt.ylabel('Fraction of positives')
            plt.xlabel('Probalities')

            plt.subplot(212)
            # Plot 2: Hist of predictions distributions
            plt.hist(self.hist_counts, histtype="step", bins=self.bins, label="Class 1", lw=2)
            plt.ylabel('Count')
            plt.xlabel('Mean predicted value')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            output_path = os.path.join(self.output_path, name_plot)
            plt.savefig(output_path)
            return output_path

    def plot_dataset_metrics_histograms(self, name_plot='hists_metrics.png'):
        """
        Plot (html/md output type) or export (json type) data on the metrics histograms.

        Parameters
        ----------
        name_plot : str, optional
            Name to give to the output plot, by default 'multiclass_hists_metrics.png'

        Returns
        -------
        str
            Output path where an image with the plot will be created.
        """
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        n_plot = len(self.metrics_names[:-1])
        n_cols = 3
        n_rows = ((n_plot - 1) // n_cols) + 1

        hists_metrics = {}
        for metric in self.metrics_names[:-1]:
            hists_metrics[metric] = np.histogram(list(self.df_dataset.loc[:, metric]), bins=bins)[0].tolist()

        if self.output_type == 'json':
            self.dict_export['df_dataset'] = self.df_dataset.to_dict()
            self.dict_export['hists metrics'] = hists_metrics
        else:
            plt.figure(figsize=(7 * n_cols, 6 * n_rows))
            for i, metric in enumerate(self.metrics_names[:-1]):
                values = hists_metrics[metric]
                plt.subplot(n_rows, n_cols, i+1)
                c = [float(i) / float(n_plot), 0.0, float(n_plot-i) / float(n_plot)]
                plt.bar(range(len(values)), values, width=0.8, linewidth=2, capsize=20, color=c)
                plt.xticks(range(len(self.bins)), bins)
                plt.title(f'{metric}', fontsize=13)
                plt.xlabel("Values bins")
                plt.grid()
                plt.ylabel("Samples count")
            plt.tight_layout(pad=3)

            output_path = os.path.join(self.output_path, name_plot)
            plt.savefig(output_path)
            return output_path

    def export_metrics_per_patch_csv(self):
        """
            Export the metrics per patch in a csv file.
        """
        path_csv = os.path.join(self.output_path, 'metrics_per_patch.csv')
        self.df_dataset.to_csv(path_csv, index=False)
