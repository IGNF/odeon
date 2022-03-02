"""
Class to manage metrics in the multiclass case.
Will generate the micro, macro and class strategies and calculate for each of them the associated metrics.
"""

import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from odeon.commons.metric.metrics import Metrics, get_metrics_from_obs, DEFAULTS_VARS
from odeon.commons.metric.plots import plot_hists


def torch_metrics_from_cm(cm_macro, class_labels):
    """
    Function to get metrics from a confusion matrix.

    Parameters
    ----------
    cm_macro : torch.Tensor
        Confusion matrix in macro strategy.

    class_labels: list
        List of string for the name of each class.

    Returns
    -------
    dict
        Metrics (per class, micro, macro) and cms (macro, micro).
    """
    metrics_collection = {"cm_macro": cm_macro}

    stats_macro = torch.zeros(cm_macro.shape[0], 4).to(cm_macro.device)

    stats_macro[:, 0] = torch.diag(cm_macro)                            # TP
    stats_macro[:, 1] = cm_macro.sum(0) - torch.diag(cm_macro)          # FP
    stats_macro[:, 2] = cm_macro.sum(1) - torch.diag(cm_macro)          # FN
    stats_macro[:, 3] = cm_macro.sum() - (stats_macro[:, 0:3].sum(1))   # TN

    # Micro 
    cm_micro = stats_macro.sum(0)
    metrics_micro = get_metrics_from_obs(true_pos = cm_micro[0],
                                         false_pos = cm_micro[1],
                                         false_neg = cm_micro[2],
                                         true_neg = cm_micro[3],
                                         micro=True)

    metrics_collection["cm_micro"] = cm_micro.reshape(2, 2)
    metrics_collection["Overall/Accuracy"] = metrics_micro["OA"] * 100
    metrics_collection["Overall/Precision"] = metrics_micro["IoU"] * 100

    # Per classes and macro
    metrics_by_class = get_metrics_from_obs(true_pos = stats_macro[:, 0],
                                            false_pos = stats_macro[:, 1],
                                            false_neg = stats_macro[:, 2],
                                            true_neg = stats_macro[:, 3])

    for metric_name, metric_per_class in metrics_by_class.items():
        metrics_collection["Average/" + metric_name] = metric_per_class.mean() * 100
        for class_idx, class_label in enumerate(class_labels):
            metrics_collection[class_label + "/" + metric_name] = metric_per_class[class_idx] * 100
    
    stats_macro = stats_macro.detach()

    return metrics_collection


def get_metrics_from_cm(cm_macro, nbr_class, class_labels, weighted, weights):
    """
    Function to get metrics from a confusion matrix.

    Parameters
    ----------
    cm_macro : np.array
        Confusion matrix in macro strategy.
    Returns
    -------
    (dict, dict, dict, np.array, np.array)
        Metrics (per class, micro, macro) and cms (per class, micro).
    """

    def get_obs_by_class_from_cm(conf_mat, class_labels):
        """
        Function to get the metrics for each class from a confusion matrix.

        Parameters
        ----------
        cm : np.array
            Input confusion matrix.

        Returns
        -------
        dict
            Dict with metrics for each class.
            The keys of th dict will be the labels of the classes.
        """
        obs_by_class = {}
        for i, class_i in enumerate(class_labels):
            obs_by_class[class_i] = {'tp': conf_mat[i, i],
                                     'fn': np.sum(conf_mat[i, :]) - conf_mat[i, i],
                                     'fp': np.sum(conf_mat[:, i]) - conf_mat[i, i],
                                     'tn': np.sum(conf_mat) - np.sum(conf_mat[i, :])
                                     - np.sum(conf_mat[:, i]) + conf_mat[i, i]}
        return obs_by_class

    obs_by_class = get_obs_by_class_from_cm(conf_mat=cm_macro,
                                            class_labels=class_labels)
    cms_classes = np.zeros([nbr_class, 2, 2])

    metrics_by_class = {}
    for i, class_i in enumerate(class_labels):
        cms_classes[i] = np.array([[obs_by_class[class_i]['tp'], obs_by_class[class_i]['fn']],
                                  [obs_by_class[class_i]['fp'], obs_by_class[class_i]['tn']]])
        metrics_by_class[class_i] = get_metrics_from_obs(obs_by_class[class_i]['tp'],
                                                         obs_by_class[class_i]['fn'],
                                                         obs_by_class[class_i]['fp'],
                                                         obs_by_class[class_i]['tn'])

    # If weights are used, the sum of the confusion matrices of each class weighted by the input weights.
    # If not, the confusions matrices of classes will directly added together.
    if weighted:
        cm_micro = np.zeros([2, 2])
        for k, weight in zip(range(nbr_class), weights):
            cm_micro += cms_classes[k] * weight
    else:
        cm_micro = np.sum(cms_classes, axis=0)

    metrics_micro = get_metrics_from_obs(cm_micro[0][0],
                                         cm_micro[0][1],
                                         cm_micro[1][0],
                                         cm_micro[1][1])

    return metrics_by_class, metrics_micro, cms_classes, cm_micro


class MetricsMulticlass(Metrics):
    """
    Class to compute metrics for the multiclass case.
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
                 get_hists_per_metrics=DEFAULTS_VARS['get_hists_per_metrics']):
        """
        This method does the same as Metrics init function :func:`~odeon.commons.metrics.Metrics.__init__()`
        """
        super().__init__(dataset=dataset,
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
                         get_hists_per_metrics=get_hists_per_metrics)

        self.cm_macro, self.cms_classes, self.cm_micro = None, None, None
        self.metrics_by_class, self.metrics_micro, self.cms_one_class = None, None, None
        self.vect_classes = {}
        self.df_report_classes, self.df_report_micro, self.df_report_macro = self.create_data_for_metrics()

        # Dicts to store info for calibrations curves.
        self.dict_prob_true, self.dict_prob_pred = {}, {}
        self.dict_hist_counts = {class_i: np.zeros(len(self.bins) - 1) for class_i in self.class_labels}
        self.dict_bin_sums = {class_i: np.zeros(len(self.bins)) for class_i in self.class_labels}
        self.dict_bin_true = {class_i: np.zeros(len(self.bins)) for class_i in self.class_labels}
        self.dict_bin_total = {class_i: np.zeros(len(self.bins)) for class_i in self.class_labels}
        self.counts_sample_per_class = {class_i: 0 for class_i in self.class_labels}

        # Dataframe for metrics per patch.
        if self.get_metrics_per_patch:
            self.header = ['name_file'] + \
                        ['OA', 'micro_IoU'] + \
                        ['macro_' + name_column for name_column in ['Precision', 'Recall', 'F1-Score', 'IoU']] + \
                        ['mean_' + name_column for name_column in self.metrics_names[:-1]] + \
                        ['_'.join(class_i.split(' ')) + '_' + name_column for class_i in self.class_labels
                         for name_column in self.metrics_names[:-1]]
            self.df_dataset = pd.DataFrame(index=range(len(self.dataset)), columns=self.header)

        if self.get_hists_per_metrics:
            self.hists_metrics = {}
            if self.output_type == 'json':
                self.dict_export['df_dataset'] = None
                self.dict_export['hists metrics'] = None

    def run(self):
        """
        Run the methods to compute metrics.
        """
        # Computed confusion matrix for each sample and sum the total in one micro cm.
        self.scan_dataset()

        # Get metrics for each strategy and cms macro from the micro cm.
        self.metrics_by_class, self.metrics_micro, self.cms_classes, self.cm_micro = \
            get_metrics_from_cm(self.cm_macro,
                                nbr_class=self.nbr_class,
                                class_labels=self.class_labels,
                                weighted=self.weighted,
                                weights=self.weights)
        # Put the calculated metrics in dataframes for reports and also computed mean metrics.
        self.metrics_to_df_reports()

        # Export values if the report is of type JSON or if roc/pr values are requested.
        self.export_values()

    def create_data_for_metrics(self):
        """
        Create dataframes to store metrics for each strategy.

        Returns
        -------
        Tuple of pd.DataFrame
            Dataframes to store metrics for each strategy.
        """
        df_report_classes = pd.DataFrame(index=self.class_labels,
                                         columns=self.metrics_names[:-1])
        df_report_micro = pd.DataFrame(index=['Values'],
                                       columns=['OA', 'IoU'])
        if self.weighted:
            df_report_macro = pd.DataFrame(index=['Average', 'Weighted avg', 'User weighted avg'],
                                           columns=['Precision', 'Recall', 'F1-Score', 'IoU'])
        else:
            df_report_macro = pd.DataFrame(index=['Average', 'Weighted avg'],
                                           columns=['Precision', 'Recall', 'F1-Score', 'IoU'])
        return df_report_classes, df_report_micro, df_report_macro

    def scan_dataset(self):
        """
        Function allowing to make a pass on the dataset in order to obtain micro confusion
        matrices by sample according to given thresholds. For each threshold, then the cms
        are added together to make only one micro cm by threshold. The function only return the
        micro cm with a threshold equal to the parameter threshold given as input.

        Returns
        -------
        np.array
            Confusion matrix for micro strategy.
        """
        self.cm_macro = np.zeros([self.nbr_class, self.nbr_class])
        # Dict and dataframe to store data for ROC and PR curves.
        self.cms_one_class = pd.DataFrame(index=self.threshold_range, columns=self.class_labels, dtype=object)
        self.cms_one_class = self.cms_one_class.applymap(lambda x: np.zeros([2, 2]))

        for index_sample, sample in enumerate(tqdm(self.dataset, desc='Metrics processing time', leave=True)):
            mask, pred, name_file = sample['mask'], sample['pred'], sample['name_file']

            if not self.in_prob_range:
                pred = self.to_prob_range(pred)

            # Compute cm micro for every sample and stack the results to a total micro cm.
            # Here binarization with an argmax.
            mask_macro, pred_macro = self.binarize(type_classifier=self.type_classifier,
                                                   prediction=pred,
                                                   mask=mask)

            # Get a cm for every sample
            conf_mat = self.get_confusion_matrix(mask_macro.flatten(),
                                                 pred_macro.flatten(),
                                                 nbr_class=self.nbr_class,
                                                 revert_order=False)
            self.cm_macro += conf_mat

            # Compute metrics per patch
            if self.get_metrics_per_patch or self.get_hists_per_metrics:
                self.compute_metrics_per_patch(conf_mat, index_sample, name_file)

            for i, class_i in enumerate(self.class_labels):
                class_mask = mask[:, :, i]
                class_pred = pred[:, :, i]

                # Stats to compute global average metrics (metrics per number of pixels).
                self.counts_sample_per_class[class_i] += np.count_nonzero(class_mask)

                if self.get_ROC_PR_curves or self.get_ROC_PR_values:
                    for threshold in self.threshold_range:
                        bin_pred = self.binarize('binary', class_pred, threshold=threshold)
                        cr_cm = self.get_confusion_matrix(class_mask.flatten(), bin_pred.flatten(),
                                                          nbr_class=2, revert_order=True)
                        self.cms_one_class.loc[threshold, class_i] += cr_cm

                if self.get_calibration_curves:
                    # To compute only once per class calibration curves.
                    # Bincounts for histogram of prediction
                    self.dict_hist_counts[class_i] += np.histogram(class_pred.flatten(), bins=self.bins)[0]
                    # Indices of the bins where the predictions will be in there.
                    binids = np.digitize(class_pred.flatten(), self.bins) - 1
                    # Bins counts of indices times the values of the predictions.
                    self.dict_bin_sums[class_i] += np.bincount(binids, weights=class_pred.flatten(),
                                                               minlength=len(self.bins))
                    # Bins counts of indices times the values of the masks.
                    self.dict_bin_true[class_i] += np.bincount(binids, weights=class_mask.flatten(),
                                                               minlength=len(self.bins))
                    # Total number observation per bins.
                    self.dict_bin_total[class_i] += np.bincount(binids, minlength=len(self.bins))

        if self.get_calibration_curves:
            for class_j in self.class_labels:
                nonzero = self.dict_bin_total[class_j] != 0  # Avoid to display null bins.
                # The proportion of samples whose class is the positive class, in each bin (fraction of positives).
                prob_true = self.dict_bin_true[class_j][nonzero] / self.dict_bin_total[class_j][nonzero]
                # The mean predicted probability in each bin.
                prob_pred = self.dict_bin_sums[class_j][nonzero] / self.dict_bin_total[class_j][nonzero]
                self.dict_prob_true[class_j] = prob_true
                self.dict_prob_pred[class_j] = prob_pred

            # Normalize dict_hist_counts to put the values between 0 and 1:
            total_pixel = np.sum(self.dict_hist_counts[self.class_labels[0]])
            self.dict_hist_counts = {key: value / total_pixel for key, value in self.dict_hist_counts.items()}

        if self.get_ROC_PR_curves or self.get_ROC_PR_values:
            self.cms_one_class = \
                self.cms_one_class.applymap(lambda cm_one_class: get_metrics_from_obs(cm_one_class[0][0],
                                                                                      cm_one_class[0][1],
                                                                                      cm_one_class[1][0],
                                                                                      cm_one_class[1][1]))
            for class_j in self.class_labels:
                vects = {'Recall': [], 'FPR': [], 'Precision': []}
                for metrics_raw in self.cms_one_class.loc[:, class_j]:
                    vects['Recall'].append(metrics_raw['Recall'])
                    vects['FPR'].append(metrics_raw['FPR'])
                    vects['Precision'].append(metrics_raw['Precision'])
                self.vect_classes[class_j] = vects

        if self.get_hists_per_metrics:
            for metric in self.header[1:]:
                self.hists_metrics[metric] = np.histogram(list(self.df_dataset.loc[:, metric]),
                                                          bins=self.bins)[0].tolist()

    def compute_metrics_per_patch(self, conf_mat_patch, index_patch, name_patch):
        """Compute metrics for a patch thanks to the confusion matrix calculated for this patch.

        Parameters
        ----------
        conf_mat_patch : np.array
            Confusion matrix of the patch.
        index_patch : int
            Index of the patch in the dataset.
        name_patch : str
            Path to the patch.
        """
        metrics_by_class, metrics_micro,  _, _ = \
            get_metrics_from_cm(conf_mat_patch,
                                nbr_class=self.nbr_class,
                                class_labels=self.class_labels,
                                weighted=self.weighted,
                                weights=self.weights)

        self.df_dataset.loc[index_patch, 'name_file'] = name_patch
        # in micro, recall = precision = f1-score = oa
        self.df_dataset.loc[index_patch, 'OA'] = metrics_micro['Precision']
        self.df_dataset.loc[index_patch, 'micro_IoU'] = metrics_micro['IoU']
        # Per classes patch metrics
        for label in self.class_labels:
            for name_column in self.metrics_names[:-1]:
                self.df_dataset.loc[index_patch, '_'.join(label.split(' ')) + '_' + name_column] \
                    = metrics_by_class[label][name_column]
        # Mean metrics per sample
        for metric in self.metrics_names[:-1]:
            mean_metric = 0
            for class_name, weight in zip(self.class_labels, self.weights):
                mean_metric += \
                    weight * self.df_dataset.loc[index_patch, '_'.join(class_name.split(' ')) + '_' + metric]
            self.df_dataset.loc[index_patch, 'mean_' + metric] = mean_metric / self.nbr_class
            if metric in ['Precision', 'Recall', 'F1-Score', 'IoU']:
                self.df_dataset.loc[index_patch, 'macro_' + metric] = mean_metric / self.nbr_class

    def metrics_to_df_reports(self):
        """
        Put the calculated metrics in dataframes for reports and also computed mean metrics.
        """
        for class_i in self.class_labels:
            self.df_report_classes.loc[class_i] = list(self.metrics_by_class[class_i].values())[:-1]
        self.df_report_classes.loc['Overall'] = self.df_report_classes.mean()

        self.df_report_macro.loc['Average'] = [self.df_report_classes.loc['Overall', 'Precision'],
                                               self.df_report_classes.loc['Overall', 'Recall'],
                                               self.df_report_classes.loc['Overall', 'F1-Score'],
                                               self.df_report_classes.loc['Overall', 'IoU']]

        total_positive_obs = sum([value for _, value in self.counts_sample_per_class.items()])
        for metric in ['Precision', 'Recall', 'F1-Score', 'IoU']:
            weighted_avg = 0
            user_weighted_avg = 0
            for i, class_i in enumerate(self.class_labels):
                weighted_avg += self.df_report_classes.loc[class_i, metric] * self.counts_sample_per_class[class_i]

                if self.weighted:
                    user_weighted_avg += self.df_report_classes.loc[class_i, metric] * self.weights[i]

            self.df_report_macro.loc['Weighted avg', metric] = \
                np.round(weighted_avg / total_positive_obs, decimals=self.decimals)

            if self.weighted:
                self.df_report_macro.loc['User weighted avg', metric] = np.round(user_weighted_avg / self.nbr_class,
                                                                                 decimals=self.decimals)

        # In micro : micro-F1 = micro-precision = micro-recall = accuracy
        self.df_report_micro.loc['Values'] = [self.metrics_micro['Precision'],
                                              self.metrics_micro['IoU']]

    def plot_dataset_metrics_histograms(self):
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
        output_paths = {}

        # 0 is name_file and and there 2 metrics in micro => 1:(1+2)
        start = 1
        end = start + self.nbr_metrics_micro
        output_paths['micro'] = plot_hists(hists=self.hists_metrics,
                                           list_metrics=self.header[start:end],
                                           output_path=osp.join(self.output_path, 'hists_micro.png'),
                                           n_bins=self.n_bins, bins_xticks=self.bins_xticks,
                                           n_cols=2, size_col=8, size_row=5)
        # there 4 metrics in macro => :(2+4)
        start = end
        end = start + self.nbr_metrics_macro
        output_paths['macro'] = plot_hists(hists=self.hists_metrics,
                                           list_metrics=self.header[start:end],
                                           output_path=osp.join(self.output_path, 'hists_macro.png'),
                                           n_bins=self.n_bins, bins_xticks=self.bins_xticks,
                                           n_cols=4, size_col=7, size_row=6)
        # Again there 6 metrics in means 6:(6 + 6)
        start = end
        end = start + self.nbr_metrics_per_class
        output_paths['means'] = plot_hists(hists=self.hists_metrics,
                                           list_metrics=self.header[start:end],
                                           output_path=osp.join(self.output_path, 'hists_means.png'),
                                           n_bins=self.n_bins, bins_xticks=self.bins_xticks)

        header_index = end
        for class_i in self.class_labels:
            header_next_index = header_index + self.nbr_metrics_per_class
            output_paths[class_i] = plot_hists(hists=self.hists_metrics,
                                               list_metrics=self.header[header_index:header_next_index],
                                               output_path=osp.join(self.output_path,
                                                                    'hists_'+'_'.join(class_i.split(' '))+'.png'),
                                               n_bins=self.n_bins, bins_xticks=self.bins_xticks)
            header_index = header_next_index
        return output_paths

    def export_values(self):
        """
            Export the metrics computed in to a dict which will be converted in JSON file
            if the report type requested is JSON. This function allow also to export the values
            computed to create ROC/PR curves in a .csv file.
        """
        if self.nbr_metrics_per_class:
            self.export_metrics_per_patch_csv()

        if self.get_ROC_PR_values:
            path_roc_csv = osp.join(self.output_path, 'ROC_PR_values.csv')
            data = {}
            data['Thresholds'] = self.threshold_range
            for class_i in self.class_labels:
                data[class_i + '_tpr'] = self.vect_classes[class_i]['Recall']
                data[class_i + '_fpr'] = self.vect_classes[class_i]['FPR']
                data[class_i + '_precision'] = self.vect_classes[class_i]['Precision']
                data[class_i + '_recall'] = self.vect_classes[class_i]['Recall']
            df_roc_pr_values = pd.DataFrame(data=data)
            df_roc_pr_values.to_csv(path_roc_csv, index=False)

        if self.output_type == 'json':
            self.dict_export['cm micro'] = self.cm_micro.tolist()
            self.dict_export['cm macro'] = self.cm_macro.tolist()
            self.dict_export['report macro'] = self.df_report_macro.T.to_dict()
            self.dict_export['report micro'] = self.df_report_micro.T.to_dict()
            self.dict_export['report classes'] = self.df_report_classes.T.to_dict()

            if self.get_ROC_PR_curves or self.get_ROC_PR_values:
                self.dict_export['PR ROC info'] = self.vect_classes

            if self.get_calibration_curves:
                dict_prob_true, dict_prob_pred, dict_hist_counts = {}, {}, {}
                for class_i in self.class_labels:
                    dict_prob_true[class_i] = self.dict_prob_true[class_i].tolist()
                    dict_prob_pred[class_i] = self.dict_prob_pred[class_i].tolist()
                    dict_hist_counts[class_i] = self.dict_hist_counts[class_i].tolist()
                self.dict_export['calibration curve'] = {'prob_true': dict_prob_true,
                                                         'prob_pred': dict_prob_pred,
                                                         'hist_counts': dict_hist_counts}

            if self.get_hists_per_metrics:
                self.dict_export['df_dataset'] = self.df_dataset.to_dict()
                self.dict_export['hists metrics'] = self.hists_metrics
