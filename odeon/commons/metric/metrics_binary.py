"""
Class to manage metrics in the binary case.
Will compute: metrics (Precision, Recall, Specifity, F1-Score, IoU), confusion matrix (cm),
ROC/PR curves, calibration curve and metrics histograms.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from odeon.commons.metric.metrics import Metrics, get_metrics_from_obs, DEFAULTS_VARS

FIGSIZE = (8, 6)


class MetricsBinary(Metrics):
    """
    Class to compute metrics for the binary case.
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
        This method does the same as Metrics init function :func:`~odeon.commons.metrics.Metrics.__init__()`
        """
        super().__init__(dataset,
                         output_path=output_path,
                         type_classifier=type_classifier,
                         in_prob_range=in_prob_range,
                         output_type=output_type,
                         class_labels=class_labels,
                         mask_bands=mask_bands,
                         pred_bands=pred_bands,
                         weights=weights,
                         threshold=threshold,
                         n_thresholds=n_thresholds,
                         bit_depth=bit_depth,
                         bins=bins,
                         n_bins=n_bins,
                         get_normalize=get_normalize,
                         get_metrics_per_patch=get_metrics_per_patch,
                         get_ROC_PR_curves=get_ROC_PR_curves,
                         get_ROC_PR_values=get_ROC_PR_values,
                         get_calibration_curves=get_calibration_curves,
                         get_hists_per_metrics=get_hists_per_metrics,
                         decimals=decimals)

        self.df_thresholds, self.cms, self.df_report_metrics = self.create_data_for_metrics()
        self.vect_curves = {}
        if self.get_metrics_per_patch or self.get_hists_per_metrics:
            self.hists_metrics = {}
            self.df_dataset = pd.DataFrame(index=range(len(self.dataset)),
                                           columns=(['name_file'] + self.metrics_names[:-1]))
        if self.get_calibration_curves:
            self.prob_true, self.prob_pred = None, None
            self.hist_counts = np.zeros(len(self.bins) - 1)

    def run(self):
        """
        Run the methods to compute metrics.
        Scan the dataset, for every threshold given the threshold range (for ROC and PR curves), for each
        sample to compute confusion matrices (cms) and metrics.
        Once the metrics and cms are computed they are exported in an output file that can have a form json,
        markdown or html. Optionally the tool can output metrics per patch and return the result as a csv file.
        """
        self.get_metrics_by_threshold()
        self.export_values()

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
        cms = {threshold: np.zeros([self.nbr_class, self.nbr_class]) for threshold in self.threshold_range}
        df_report_metrics = pd.DataFrame(index=['Values'], columns=self.metrics_names[:-1])
        return df_thresholds, cms, df_report_metrics

    def get_metrics_by_threshold(self):
        """
        Function allowing to make a pass on the dataset in order to obtain confusion
        matrices by sample according to given thresholds. For each threshold, then the cms
        are added together to make only one cm by threshold.
        """
        bin_sums = np.zeros(len(self.bins))
        bin_true = np.zeros(len(self.bins))
        bin_total = np.zeros(len(self.bins))

        for dataset_index, sample in enumerate(tqdm(self.dataset, desc='Metrics processing time', leave=True)):
            mask, pred, name_file = sample['mask'], sample['pred'], sample['name_file']

            if not self.in_prob_range:
                pred = self.to_prob_range(pred)

            # Compute metrics per patch and return an histogram of the values
            if self.get_metrics_per_patch or self.get_hists_per_metrics:
                pred_cm = self.binarize(self.type_classifier, pred, threshold=self.threshold)
                conf_mat = self.get_confusion_matrix(mask.flatten(), pred_cm.flatten(),
                                                     self.nbr_class, revert_order=True)

                sample_metrics = self.get_metrics_from_cm(conf_mat)
                self.df_dataset.loc[dataset_index, 'name_file'] = name_file
                for name_column in self.metrics_names[:-1]:
                    self.df_dataset.loc[dataset_index, name_column] = sample_metrics[name_column]

            # To calcultate info for calibrations curves only once.
            if self.get_calibration_curves:
                # bincounts for histogram of prediction
                self.hist_counts += np.histogram(pred.flatten(), bins=self.bins)[0]
                # Indices of the bins where the predictions will be in there.
                binids = np.digitize(pred.flatten(), self.bins) - 1
                # Bins counts of indices times the values of the predictions.
                bin_sums += np.bincount(binids, weights=pred.flatten(), minlength=len(self.bins))
                # Bins counts of indices times the values of the masks.
                bin_true += np.bincount(binids, weights=mask.flatten(), minlength=len(self.bins))
                # Total number observation per bins.
                bin_total += np.bincount(binids, minlength=len(self.bins))

            for threshold in self.threshold_range:
                pred_cm = self.binarize(self.type_classifier, pred, threshold=threshold)
                conf_mat = self.get_confusion_matrix(mask.flatten(), pred_cm.flatten(),
                                                     self.nbr_class, revert_order=True)
                self.cms[threshold] += conf_mat

        # Compute metrics at every threshold value once the whole dataset has been covered.
        for threshold in self.threshold_range:
            cr_metrics = self.get_metrics_from_cm(self.cms[threshold])
            for metric, metric_value in cr_metrics.items():
                self.df_thresholds.loc[self.df_thresholds['threshold'] == threshold, metric] = metric_value

        if self.get_calibration_curves:
            nonzero = bin_total != 0  # Avoid to display null bins.
            self.prob_true = bin_true[nonzero] / bin_total[nonzero]
            self.prob_pred = bin_sums[nonzero] / bin_total[nonzero]

        if self.get_hists_per_metrics:
            for metric in self.metrics_names[:-1]:
                self.hists_metrics[metric] = np.histogram(list(self.df_dataset.loc[:, metric]),
                                                          bins=self.bins)[0].tolist()

        if self.get_ROC_PR_curves or self.get_ROC_PR_values:
            self.vect_curves = {"TPR": self.df_thresholds['Recall'].tolist(),
                                "FPR": self.df_thresholds['FPR'].tolist(),
                                "Precision": self.df_thresholds['Precision'].tolist(),
                                "Recall": self.df_thresholds['Recall'].tolist()}

        # Put in the df for the report the computed metrics for the threshold pass as input in the configuration file.
        self.df_report_metrics.loc['Values'] = \
            self.df_thresholds.loc[self.df_thresholds['threshold'] == self.threshold, self.metrics_names[:-1]].values

    def get_metrics_from_cm(self, conf_mat):
        """
        Extract the metrics from a cm.

        Parameters
        ----------
        conf_mat : np.array[type]
            Confusion matrix in micro strategy.

        Returns
        -------
        dict
            Dict with the metrics.
        """
        true_pos, false_neg, false_pos, true_neg = conf_mat.ravel()
        return get_metrics_from_obs(true_pos, false_neg, false_pos, true_neg)

    def export_values(self):
        """
            Export the values used to create a report.
        """
        if self.get_metrics_per_patch:
            self.export_metrics_per_patch_csv()

        if self.get_ROC_PR_values:
            path_roc_csv = os.path.join(self.output_path, 'ROC_PR_values.csv')
            df_roc_pr_values = pd.DataFrame(data={"Thresholds": self.threshold_range,
                                                  "TPR": self.df_thresholds['Recall'],
                                                  "FPR": self.df_thresholds['FPR'],
                                                  "Precision": self.df_thresholds['Precision'],
                                                  "Recall": self.df_thresholds['Recall']})
            df_roc_pr_values.to_csv(path_roc_csv, index=False)

        if self.output_type == 'json':
            if self.get_calibration_curves:
                self.dict_export['calibration curve'] = {'prob_true': self.prob_true.tolist(),
                                                         'prob_pred': self.prob_pred.tolist(),
                                                         'hist_counts': self.hist_counts.tolist()}

            if self.get_ROC_PR_curves or self.get_ROC_PR_values:
                self.dict_export['roc_curve'] = self.vect_curves

            if self.get_hists_per_metrics:
                self.dict_export['df_dataset'] = self.df_dataset.to_dict()
                self.dict_export['hists metrics'] = self.hists_metrics
