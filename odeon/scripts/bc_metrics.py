import numpy as np
import pandas as pd
from PIL import Image
from metrics import Metrics, DEFAULTS_VARS


class BC_Metrics(Metrics):

    def __init__(self,
                 mask_files,
                 pred_files,
                 output_path,
                 threshold,
                 bit_depth=DEFAULTS_VARS['bit_depth'],
                 roc_range=DEFAULTS_VARS['roc_range'],
                 batch_size=DEFAULTS_VARS['batch_size'],
                 num_workers=DEFAULTS_VARS['num_workers']):

        super().__init__(mask_files=mask_files,
                         pred_files=pred_files,
                         output_path=output_path,
                         bit_depth=bit_depth,
                         batch_size=batch_size,
                         num_workers=num_workers)

        self.nbr_class = 2  # Crée un moyen de récupérer proprement le nombre de classes.
        self.obs_names = ['tp', 'fn', 'fp', 'tn']
        self.metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'IoU', 'FPR']
        self.cms = np.zeros((self.nbr_class, self.nbr_class))
        self.threshold = threshold
        self.roc_range = roc_range
        self.type_prob, self.in_prob_range = self.get_info_pred()
        self.df_dataset, self.df_roc_metrics, self.cms = self.create_data_for_metrics()
        self.get_metrics_ROC()
        self.plot_confusion_matrix(self.cms[self.threshold])
        self.plot_ROC_curve(self.df_roc_metrics['FPR'], self.df_roc_metrics['Recall'], generate=False)
        self.plot_PR_curve(self.df_roc_metrics['Precision'], self.df_roc_metrics['Recall'], generate=False)

    def get_info_pred(self):
        """
            Make test on the first tenth of the predictions to check if inputs preds are in soft or in hard.
        """
        pred_samples = self.pred_files[:len(self.pred_files)//10]
        if len(pred_samples) == 1:
            pred_samples = list(pred_samples)
        nuniques = 0
        maxu = - float('inf')
        for path_pred in pred_samples:
            with Image.open(path_pred) as pred:
                pred = np.array(pred)
                cr_nuniques = np.unique(pred.flatten())
                if len(cr_nuniques) > nuniques:
                    nuniques = len(cr_nuniques)
                if max(cr_nuniques) > maxu:
                    maxu = max(cr_nuniques)

        type_prob = 'soft' if nuniques > 2 else 'hard'
        in_prob_range = True if maxu <= 1 else False
        return type_prob, in_prob_range

    def to_prob_range(self, value):
        return value / self.depth_dict[self.bit_depth]

    def create_data_for_metrics(self):
        df_dataset = pd.DataFrame(index=range(len(self.pred_files)), columns=['pred', 'mask'] + self.metrics_names)
        df_dataset['pred'] = self.pred_files
        df_dataset['mask'] = self.mask_files

        df_roc_metrics = pd.DataFrame(index=range(len(self.roc_range)),
                                      columns=(['threshold'] + self.obs_names + self.metrics_names))
        df_roc_metrics['threshold'] = self.roc_range
        cms = {}
        return df_dataset, df_roc_metrics, cms

    def binarize(self, prediction, threshold):
        if not self.in_prob_range:
            tmp = np.array(prediction, dtype=np.float32) / 255  # in case of pred in unint8 format.
        tmp[tmp > threshold] = 1
        tmp[tmp <= threshold] = 0
        return tmp

    def get_metrics_by_sample(self):
        for mask_file, pred_file in zip(self.mask_files, self.pred_files):
            with Image.open(mask_file) as mask_img:
                mask = np.array(mask_img)
            with Image.open(pred_file) as pred_img:
                pred = np.array(pred_img)

            pred = self.binarize(pred, self.threshold)
            cm = self.get_confusion_matrix(mask.flatten(), pred.flatten())
            self.cms += cm
            cr_metrics = self.get_metrics_from_cm(cm)

            for metric_name in self.metrics_names:
                self.df_dataset.loc[self.df_dataset['pred'] == pred_file, metric_name] = cr_metrics[metric_name]

    def get_metrics_ROC(self):

        for threshold in self.roc_range:

            threshold_metrics = {name: 0 for name in (self.obs_names + self.metrics_names)}

            for mask_file, pred_file in zip(self.mask_files, self.pred_files):

                with Image.open(mask_file) as mask_img:
                    mask = np.array(mask_img)

                with Image.open(pred_file) as pred_img:
                    pred = np.array(pred_img)
                pred = self.binarize(pred, threshold)
                cm = self.get_confusion_matrix(mask.flatten(), pred.flatten())
                self.cms[threshold] = cm
                cr_metrics = self.get_metrics_from_cm(cm)
                threshold_metrics.update({
                    name: threshold_metrics[name] + cr_metrics[name] for name in (self.obs_names + self.metrics_names)})

            for metric, metric_value in threshold_metrics.items():
                self.df_roc_metrics.loc[self.df_roc_metrics['threshold'] == threshold, metric] = \
                    metric_value / len(self.mask_files)

    def get_confusion_matrix(self, prediction, target):
        """Returns the confusion matrix for one class or threshold.
        TP (true positives): the number of correctly classified pixels (1 -> 1)
        TN (true negatives): the number of correctly not classified pixels (0 -> 0)
        FP (false positives): the number of pixels wrongly classified (0 -> 1)
        FN (false negatives): the number of pixels wrongly not classifed (1 -> 0)

        Parameters
        ----------
        prediction : ndarray
            binary inference
        target : ndarray
            binary ground truth

        Returns
        -------
        ndarray
            confusion matrix [[TP,FN],[FP,TN]]
        """
        tp = np.sum(np.logical_and(target, prediction))
        tn = np.sum(np.logical_not(np.logical_or(target, prediction)))
        fp = np.sum(prediction[target == 0] == 1)
        fn = np.sum(prediction[target == 1] == 0)
        return np.array([[tp, fn], [fp, tn]])

    def get_metrics_from_cm(self, cm):
        tp, fn, fp, tn = cm.ravel()

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

        # False Positive Rate (FPR)
        if fp != 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0

        return {'tp': tp,
                'fn': fn,
                'fp': fp,
                'tn': tn,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'F1-Score': f1_score,
                'IoU': iou,
                'FPR': fpr}

