"""
Class to manage metrics in the multiclass case.
Will generate the micro, macro and class strategies and calculate for each of them the associated metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from odeon.commons.metrics.metrics import Metrics, DEFAULTS_VARS
from tqdm import tqdm


class Metrics_Multiclass(Metrics):

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
        Initialize the class attributes and create the dataframes to store the metrics for each strategy.
        Scan the dataset, by threshold (for ROC and PR curves in per class strategy), by classes and
        finally by sample to compute confusion matrices (cms) and metrics.
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
            Here the classfier type should be 'multiclass'.
        output_type : str, optional
            Desired format for the output file. Could be json, md or html.
            A report will be created if the output type is html or md.
            If the output type is json, all the data will be exported in a dict in order
            to be easily reusable, by default html.
        class_labels : list of str, optional
            Label for each class in the dataset.
            If None the labels of the classes will be of type: class 1, class 2, etc .., by default None
        weights : list of number, optional
            List of weights to balance the metrics.
            Used for the macro matrix and the mean metrics, by default None.
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

        super().__init__(dataset=dataset,
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

        self.df_report_classes, self.df_report_micro, self.df_report_macro = self.create_data_for_metrics()

        # Dataframe for metrics per patch.
        if self.get_metrics_per_patch:
            self.header = ['name_file'] + \
                        ['macro_' + name_column for name_column in self.metrics_names[:-1]] + \
                        ['micro_' + name_column for name_column in ['Precision', 'Recall', 'F1-Score', 'IoU']] + \
                        ['mean_' + name_column for name_column in self.metrics_names[:-1]] + \
                        ['_'.join(class_i.split(' ')) + '_' + name_column for class_i in self.class_labels
                         for name_column in self.metrics_names[:-1]]
            self.df_dataset = pd.DataFrame(index=range(len(self.dataset)), columns=self.header)

        # Computed confusion matrix for each sample and sum the total in one micro cm.
        self.cm_micro = self.scan_dataset()

        # Get metrics for each strategy and cms macro from the micro cm.
        self.metrics_by_class, self.metrics_micro, self.metrics_macro, self.cms_classes, self.cm_macro = \
            self.get_metrics_from_cm(self.cm_micro)

        # Put the calculated metrics in dataframes for reports and also computed mean metrics.
        self.metrics_to_df_reports()

        # Create a csv to export the computed metrics per patch.
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
        df_report_classes = pd.DataFrame(index=[class_name for class_name in self.class_labels],
                                         columns=self.metrics_names[:-1])
        df_report_micro = pd.DataFrame(index=['Values'],
                                       columns=['Precision', 'Recall', 'F1-Score', 'IoU'])
        df_report_macro = pd.DataFrame(index=['Values'],
                                       columns=self.metrics_names[:-1])
        return df_report_classes, df_report_micro, df_report_macro

    def scan_dataset(self):
        """
        Function allowing to make a pass on the dataset in order to obtain micro confusion
        matrices by sample according to given thresholds. For each threshold, then the cms
        are added together to make only micro cm by threshold. The function only return the
        micro cm with a threshold equal to the parameter threshold given as input.

        Returns
        -------
        np.array
            Confusion matrix for micro strategy.
        """
        cm_micro = np.zeros([self.nbr_class, self.nbr_class])
        # Dict to store info for ROC and PR curves.
        self.vect_classes = {}
        # Dicts to store info for claibrations curves.
        self.dict_hist_counts, self.dict_prob_true, self.dict_prob_pred = {}, {}, {}

        for i, class_i in tqdm(enumerate(self.class_labels), desc='Metrics processing time', leave=True):
            # Dict for PR and ROC curves.
            vects = {'Recall': [],
                     'FPR': [],
                     'Precision': []}

            # Data for calibration curves.
            hist_counts = np.zeros(len(self.bins) - 1)
            bin_sums = np.zeros(len(self.bins))
            bin_true = np.zeros(len(self.bins))
            bin_total = np.zeros(len(self.bins))

            for threshold in tqdm(self.threshold_range, desc=class_i, leave=True):
                cms_one_class = np.zeros([2, 2])

                dataset_index = 0
                for sample in self.dataset:
                    mask, pred, name_file = sample['mask'], sample['pred'], sample['name_file']

                    class_mask = mask.copy()[:, :, i]
                    class_pred = pred.copy()[:, :, i]
                    bin_pred = self.binarize('binary', class_pred, threshold=threshold)
                    cr_cm = self.get_confusion_matrix(class_mask.flatten(), bin_pred.flatten(), nbr_class=2)
                    cms_one_class += cr_cm

                    # To compute only once data for cm micro.
                    if threshold == self.threshold and i == 0:
                        # Compute cm micro for every sample and stack the results to a total micro cm.
                        mask_micro, pred_micro = self.binarize(self.type_classifier, pred.copy(), mask=mask)
                        cm = self.get_confusion_matrix(mask_micro.flatten(), pred_micro.flatten())
                        cm_micro += cm

                        # Compute metrics per patch
                        if self.get_metrics_per_patch:
                            metrics_by_class, metrics_micro, metrics_macro, _, _ = self.get_metrics_from_cm(cm)
                            self.df_dataset.loc[dataset_index, 'name_file'] = name_file
                            for name_column in self.metrics_names[:-1]:
                                self.df_dataset.loc[dataset_index, 'macro_' + name_column] = metrics_macro[name_column]
                            for name_column in ['Precision', 'Recall', 'F1-Score', 'IoU']:
                                self.df_dataset.loc[dataset_index, 'micro_' + name_column] = metrics_micro[name_column]
                            for label in self.class_labels:
                                for name_column in self.metrics_names[:-1]:
                                    self.df_dataset.loc[dataset_index, '_'.join(label.split(' ')) + '_' + name_column] \
                                        = metrics_by_class[label][name_column]

                            # Mean metrics per sample
                            for metric in self.metrics_names[:-1]:
                                mean_metric = 0
                                for class_name, weight in zip(self.class_labels, self.weights):
                                    mean_metric += \
                                        weight * self.df_dataset.loc[dataset_index, '_'.join(class_name.split(' ')) +
                                                                                    '_' + metric]
                                self.df_dataset.loc[dataset_index, 'mean_' + metric] = mean_metric / self.nbr_class
                            dataset_index += 1

                    # To compute only once per class calibration curves.
                    if threshold == self.threshold:
                        pred_hist = pred.copy()[:, :, i]
                        mask_hist = mask.copy()[:, :, i]

                        if not self.in_prob_range:
                            pred_hist = self.to_prob_range(pred_hist)

                        # Bincounts for histogram of prediction
                        hist_counts += np.histogram(pred_hist.flatten(), bins=self.bins)[0]
                        # Indices of the bins where the predictions will be in there.
                        binids = np.digitize(pred_hist.flatten(), self.bins) - 1
                        # Bins counts of indices times the values of the predictions.
                        bin_sums += np.bincount(binids, weights=pred_hist.flatten(), minlength=len(self.bins))
                        # Bins counts of indices times the values of the masks.
                        bin_true += np.bincount(binids, weights=mask_hist.flatten(), minlength=len(self.bins))
                        # Total number observation per bins.
                        bin_total += np.bincount(binids, minlength=len(self.bins))

                cr_metrics = self.get_metrics_from_obs(cms_one_class[0][0],
                                                       cms_one_class[0][1],
                                                       cms_one_class[1][0],
                                                       cms_one_class[1][1])

                # Collect info for PR/ROC curves
                vects['Recall'].append(cr_metrics['Recall'])
                vects['FPR'].append(cr_metrics['FPR'])
                vects['Precision'].append(cr_metrics['Precision'])

            self.vect_classes[class_i] = vects

            nonzero = bin_total != 0  # Avoid to display null bins.
            # The proportion of samples whose class is the positive class, in each bin (fraction of positives).
            prob_true = bin_true[nonzero] / bin_total[nonzero]
            # The mean predicted probability in each bin.
            prob_pred = bin_sums[nonzero] / bin_total[nonzero]

            self.dict_hist_counts[class_i] = hist_counts
            self.dict_prob_true[class_i] = prob_true
            self.dict_prob_pred[class_i] = prob_pred

        return cm_micro

    def get_obs_by_class_from_cm(self, cm):
        """
        Function to get from a confusion matrix the metrics for each class.

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
        for i, class_i in enumerate(self.class_labels):
            obs_by_class[class_i] = {'tp': cm[i, i],
                                     'fn': np.sum(cm[i, :]) - cm[i, i],
                                     'fp': np.sum(cm[:, i]) - cm[i, i],
                                     'tn': np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i])}
        return obs_by_class

    def get_obs_micro_from_cm(self, cm):
        """
        In micro strategy, extract the number of tp, fn, fp from a cm.

        Parameters
        ----------
        cm : np.array[type]
            Confusion matrix in micro strategy.

        Returns
        -------
        dict
            Dict with the number of observations for tp, fn, fp.
        """
        obs_micro = {}
        obs_micro = {'tp': np.sum(np.diag(cm)),
                     'fn': np.sum(np.triu(cm, k=1)),
                     'fp': np.sum(np.tril(cm, k=-1))}
        return obs_micro

    def get_metrics_from_cm(self, cm_micro):
        """
        Function to get metrics from a confusion matrix.

        Parameters
        ----------
        cm_micro : np.array
            Confusion matrix in micro strategy.
        Returns
        -------
        (dict, dict, dict, np.array, np.array)
            Metrics (macro, micro, per class) and cms (macro, per class).
        """
        obs_by_class = self.get_obs_by_class_from_cm(cm_micro)
        obs_micro = self.get_obs_micro_from_cm(cm_micro)

        cms_classes = np.zeros([self.nbr_class, 2, 2])

        metrics_by_class = {}
        for i, class_i in enumerate(self.class_labels):
            cms_classes[i] = np.array([[obs_by_class[class_i]['tp'], obs_by_class[class_i]['fn']],
                                       [obs_by_class[class_i]['fp'], obs_by_class[class_i]['tn']]])
            metrics_by_class[class_i] = self.get_metrics_from_obs(obs_by_class[class_i]['tp'],
                                                                  obs_by_class[class_i]['fn'],
                                                                  obs_by_class[class_i]['fp'],
                                                                  obs_by_class[class_i]['tn'])

        # If weights are used, the sum of the confusion matrices of each class weighted by the input weights.
        # If not, the confusions matrices of classes will directly added together.
        if self.weighted:
            cm_macro = np.zeros([2, 2])
            for k, weight in zip(range(self.nbr_class), self.weights):
                cm_macro += cms_classes[k] * weight
        else:
            cm_macro = np.sum(cms_classes, axis=0)

        # We will only look at Precision, Recall, F1-Score and IoU.
        # Others metrics will be false because we don't have any TN in micro.
        metrics_micro = self.get_metrics_from_obs(obs_micro['tp'],
                                                  obs_micro['fn'],
                                                  obs_micro['fp'],
                                                  0)

        metrics_macro = self.get_metrics_from_obs(cm_macro[0][0],
                                                  cm_macro[0][1],
                                                  cm_macro[1][0],
                                                  cm_macro[1][1])

        return metrics_by_class, metrics_micro, metrics_macro, cms_classes, cm_macro

    def metrics_to_df_reports(self):
        """
        Put the calculated metrics in dataframes for reports and also computed mean metrics.
        """
        for class_i in self.class_labels:
            self.df_report_classes.loc[class_i] = list(self.metrics_by_class[class_i].values())[:-1]
        # self.df_report_classes.loc['Overall'] = self.df_report_classes.mean()

        # If weights are not used its equal to metrics in macro strategy.
        cm_overall = np.sum(self.cms_classes, axis=0)
        metrics_overall = self.get_metrics_from_obs(cm_overall[0][0],
                                                    cm_overall[0][1],
                                                    cm_overall[1][0],
                                                    cm_overall[1][1])
        self.df_report_classes.loc['Overall'] = list(metrics_overall.values())[:-1]

        self.df_report_micro.loc['Values'] = [self.metrics_micro['Precision'],
                                              self.metrics_micro['Recall'],
                                              self.metrics_micro['F1-Score'],
                                              self.metrics_micro['IoU']]

        self.df_report_macro.loc['Values'] = list(self.metrics_macro.values())[:-1]

    def plot_ROC_PR_per_class(self, name_plot='multiclass_roc_pr_curves.png'):
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
        if self.output_type == 'json':
            self.dict_export['PR ROC info'] = self.vect_classes
        else:
            plt.figure(figsize=(16, 8))
            plt.subplot(121)
            for class_i in self.class_labels:
                fpr = np.array(self.vect_classes[class_i]['FPR'])
                tpr = np.array(self.vect_classes[class_i]['Recall'])
                fpr, tpr = np.insert(fpr, 0, 1), np.insert(tpr, 0, 1)
                fpr, tpr = np.append(fpr, 0), np.append(tpr, 0)
                fpr, tpr = fpr[::-1], tpr[::-1]
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
                precision = np.array(self.vect_classes[class_i]['Precision'])
                recall = np.array(self.vect_classes[class_i]['Recall'])
                precision = np.array([1 if p == 0 and r == 0 else p for p, r in zip(precision, recall)])
                idx = np.argsort(recall)
                recall, precision = recall[idx], precision[idx]
                recall, precision = np.insert(recall, 0, 0), np.insert(precision, 0, 1)
                recall, precision = np.append(recall, 1), np.append(precision, 0)
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, label=f'{class_i} AUC = {round(pr_auc, 3)}')
            plt.plot([1, 0], [0, 1], 'r--')
            plt.title('Precision-Recall Curve')
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.legend(loc='lower left')
            plt.grid(True)

            output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
            plt.savefig(output_path)
            return output_path

    def plot_calibration_curve(self, name_plot='multiclass_calibration_curves.png'):
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
        # Normalize dict_hist_counts to put the values between 0 and 1:
        total_pixel = np.sum(self.dict_hist_counts[self.class_labels[0]])
        self.dict_hist_counts = {key: value / total_pixel for key, value in self.dict_hist_counts.items()}

        if self.output_type == 'json':
            dict_prob_true, dict_prob_pred, dict_hist_counts = {}, {}, {}
            for class_i in self.class_labels:
                dict_prob_true[class_i] = self.dict_prob_true[class_i].tolist()
                dict_prob_pred[class_i] = self.dict_prob_pred[class_i].tolist()
                dict_hist_counts[class_i] = self.dict_hist_counts[class_i].tolist()

            self.dict_export['calibration curve'] = {'prob_true': dict_prob_true,
                                                     'prob_pred': dict_prob_pred,
                                                     'hist_counts': dict_hist_counts}
        else:
            plt.figure(figsize=(16, 8))
            # Plot 1: calibration curves
            plt.subplot(211)
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            for class_i in self.class_labels:
                plt.plot(self.dict_prob_true[class_i], self.dict_prob_pred[class_i], "s-", label=class_i)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('Calibration plots (reliability curve)')
            plt.ylabel('Fraction of positives')
            plt.xlabel('Probalities')

            # Plot 2: Hist of predictions distributions
            plt.subplot(212)
            for class_i in self.class_labels:
                plt.hist(self.dict_hist_counts[class_i], bins=self.bins, histtype="step", label=class_i, lw=2)
            plt.ylabel('Count')
            plt.xlabel('Mean predicted value')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout(pad=3)

            output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
            plt.savefig(output_path)
            return output_path

    def plot_hists(self, list_metrics, n_cols=3, size_col=5, size_row=4, bins=None, name_plot=None):
        """
        Plot metrics histograms.

        Parameters
        ----------
        list_metrics : list
            Name of the metrics to plot.
        n_cols : int, optional
            number of columns in the figure, by default 3
        size_col : int, optional
            size of a column in the figure, by default 5
        size_row : int, optional
            size of a row in the figure, by default 4
        bins : list of int, optional
            Bins used for bins counts for the creation of the histograms, by default None
        name_plot : str, optional
            Name to give to the output plot, by default 'multiclass_hists_metrics.png'

        Returns
        -------
        str
            Output path where an image with the plot will be created.
        """
        colors = plt.rcParams["axes.prop_cycle"]()

        if bins is None:
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        n_plot = len(list_metrics)
        n_rows = ((n_plot - 1) // n_cols) + 1
        plt.figure(figsize=(size_col * n_cols, size_row * n_rows))
        for i, metric in enumerate(list_metrics):
            values = self.hists_metrics[metric]
            plt.subplot(n_rows, n_cols, i+1)
            c = next(colors)["color"]
            plt.bar(range(len(values)), values, width=0.8, linewidth=2, capsize=20, color=c)
            plt.xticks(range(len(self.bins)), bins)
            plt.title(f"{' '.join(metric.split('_'))}", fontsize=13)
            plt.xlabel("Values bins")
            plt.grid()
            plt.ylabel("Samples count")

        plt.tight_layout(pad=3)
        output_path = os.path.join(os.path.dirname(self.output_path), name_plot)
        plt.savefig(output_path)
        return output_path

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
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        self.hists_metrics = {}
        for metric in self.header[1:]:
            self.hists_metrics[metric] = np.histogram(list(self.df_dataset.loc[:, metric]), bins=bins)[0].tolist()

        if self.output_type == 'json':
            self.dict_export['df_dataset'] = self.df_dataset.to_dict()
            self.dict_export['hists metrics'] = self.hists_metrics
        else:
            output_paths = {}
            output_paths['macro'] = self.plot_hists(self.header[1:7], bins=bins, name_plot='hists_macro.png')
            output_paths['micro'] = self.plot_hists(self.header[7:11], n_cols=4, size_col=7, size_row=6, bins=bins,
                                                    name_plot='hists_micro.png')
            output_paths['means'] = self.plot_hists(self.header[11:17], bins=bins, name_plot='hists_means.png')

            header_index = 17
            for class_i in self.class_labels:
                header_next_index = header_index + self.nbr_class - 1
                output_paths[class_i] = self.plot_hists(self.header[header_index:header_next_index],
                                                        bins=bins,
                                                        name_plot='hists_'+'_'.join(class_i.split(' '))+'.png')
                header_index = header_next_index
            return output_paths

    def export_metrics_per_patch_csv(self):
        """
            Export the metrics per patch in a csv file.
        """
        path_csv = os.path.join(self.output_path, 'metrics_per_patch.csv')
        self.df_dataset.to_csv(path_csv, index=False)
