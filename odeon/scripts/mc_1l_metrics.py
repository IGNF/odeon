import numpy as np
import pandas as pd
from metrics import Metrics, DEFAULTS_VARS
# from odeon.commons.metrics import Metrics


class MC_1L_Metrics(Metrics):

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

        self.df_report_classes, self.df_report_micro, self.df_report_macro = self.create_data_for_metrics()
        self.cm_micro = self.get_cm_micro()
        self.metrics_by_class, self.metrics_micro, self.metrics_macro = self.get_metrics_from_cm()

    def create_data_for_metrics(self):
        df_report_classes = pd.DataFrame(index=[class_name for class_name in self.classes_labels],
                                         columns=self.metrics_names[:-1])
        df_report_micro = pd.DataFrame(index=['Values'],
                                       columns=['Precision', 'Recall', 'F1-Score', 'IoU'])
        df_report_macro = pd.DataFrame(index=['Values'],
                                       columns=self.metrics_names[:-1])
        return df_report_classes, df_report_micro, df_report_macro

    def binarize(self, mask, pred):
        return np.argmax(mask, axis=2), np.argmax(pred, axis=2)

    def get_cm_micro(self):
        cm_micro = np.zeros([self.nbr_class, self.nbr_class])
        for mask, pred in zip(self.masks, self.preds):
            mask, pred = self.binarize(mask, pred)
            cm = self.get_confusion_matrix(mask.flatten(), pred.flatten())
            cm_micro += cm
        return cm_micro

    def get_obs_by_class_from_cm(self, cm):
        obs_by_class = {}
        for i, class_i in enumerate(self.classes):
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
        for i, class_i in enumerate(self.classes):
            self.cms_classes[i] = np.array([[obs_by_class[class_i]['tp'], obs_by_class[class_i]['fn']],
                                            [obs_by_class[class_i]['fp'], obs_by_class[class_i]['tn']]])
            metrics_by_class[class_i] = self.get_metrics_from_obs(self.cms_classes[i].ravel())

        self.cm_macro = np.sum(self.cms_classes, axis=0)

        # We will only look at Precision, Recall, F1-Score and IoU.
        # Others metrics will be false because we don't have any TN in micro.
        metrics_micro = self.get_metrics_from_obs(obs_micro['tp'],
                                                  obs_micro['fn'],
                                                  obs_micro['fp'],
                                                  0)

        metrics_macro = self.get_metrics_from_obs(self.cm_macro.ravel())

        return metrics_by_class, metrics_micro, metrics_macro
