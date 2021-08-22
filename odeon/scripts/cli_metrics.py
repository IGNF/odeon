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
    - AUC Score for ROC and PR curves
    - Calibration Curves
    - KL Divergence

* Multi-class case:
    - Metrics Per class: same metrics as the binary case for each class.
    - Macro (1 versus all then all cms stacked): same metrics as the binary case for the sum of all classes.
    - Micro : Precision, Recall, F1 Score (confusion matrix but no ROC curve).

* Multi-labels case:
    - Same as the multi-class case but without the global confusion matrix in  micro analysis.
"""
import os
import csv
from odeon import LOGGER
from odeon.commons.core import BaseTool
from odeon.commons.image import image_to_ndarray
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.nn.datasets import MetricsDataset
from metrics_factory import Metrics_Factory
from metrics import DEFAULTS_VARS


class CLI_Metrics(BaseTool):

    def __init__(self,
                 mask_path,
                 pred_path,
                 output_path,
                 type_classifier,
                 output_type=None,
                 class_labels=None,
                 threshold=DEFAULTS_VARS['threshold'],
                 threshold_range=DEFAULTS_VARS['threshold_range'],
                 bit_depth=DEFAULTS_VARS['bit_depth'],
                 nb_calibration_bins=DEFAULTS_VARS['nb_calibration_bins'],
                 batch_size=DEFAULTS_VARS['batch_size'],
                 num_workers=DEFAULTS_VARS['num_workers'],
                 normalize=DEFAULTS_VARS['normalize'],
                 get_metrics_per_patch=DEFAULTS_VARS['get_metrics_per_patch'],
                 get_ROC_PR_curves=DEFAULTS_VARS['get_ROC_PR_curves'],
                 get_calibration_curves=DEFAULTS_VARS['get_calibration_curves'],
                 get_hists_per_metrics=DEFAULTS_VARS['get_hists_per_metrics']):

        self.mask_path = mask_path
        self.pred_path = pred_path

        if not os.path.exists(output_path):
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Output folder ${output_path} does not exist.")
        elif not os.path.isdir(output_path):
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Output path ${output_path} should be a folder.")
        else:
            self.output_path = output_path

        if output_type in ['md', 'json', 'html']:
            self.output_type = output_type
        else:
            LOGGER.error('ERROR: the output file can only be in md, json, html.')
            self.output_type = 'html'

        self.type_classifier = type_classifier
        self.class_labels = class_labels,
        self.threshold = threshold
        self.threshold_range = threshold_range
        self.bit_depth = bit_depth
        self.nb_calibration_bins = nb_calibration_bins

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize

        self.get_metrics_per_patch = get_metrics_per_patch
        self.get_ROC_PR_curves = get_ROC_PR_curves
        self.get_calibration_curves = get_calibration_curves
        self.get_hists_per_metrics = get_hists_per_metrics

        self.mask_files, self.pred_files = self.get_files_from_input_paths()
        self.height, self.width, self.nbr_class = self.get_samples_shapes()

        metrics_dataset = MetricsDataset(self.mask_files,
                                         self.pred_files,
                                         nbr_class=self.nbr_class,
                                         width=self.width,
                                         height=self.height)

        self.metrics = Metrics_Factory(self.type_classifier)(dataset=metrics_dataset,
                                                             output_path=self.output_path,
                                                             output_type=self.output_type,
                                                             type_classifier=self.type_classifier,
                                                             class_labels=self.class_labels,
                                                             threshold=self.threshold,
                                                             threshold_range=self.threshold_range,
                                                             bit_depth=self.bit_depth,
                                                             nb_calibration_bins=self.nb_calibration_bins,
                                                             batch_size=self.batch_size,
                                                             num_workers=self.num_workers,
                                                             normalize=normalize,
                                                             get_metrics_per_patch=self.get_metrics_per_patch,
                                                             get_ROC_PR_curves=self.get_ROC_PR_curves,
                                                             get_calibration_curves=get_calibration_curves,
                                                             get_hists_per_metrics=get_hists_per_metrics)

    def __call__(self):
        self.metrics()

    def get_files_from_input_paths(self):
        if not os.path.exists(self.mask_path):
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Masks folder {self.mask_path} does not exist.")
        elif not os.path.exists(self.pred_path):
            LOGGER.error(f'ERROR: Predictions folder {self.pred_path} does not exist.')
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Predictions folder {self.pred_path} does not exist.")
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

    def get_samples_shapes(self):
        mask_file, pred_file = self.mask_files[0], self.pred_files[0]

        if not os.path.exists(mask_file):
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"File ${mask_file} does not exist.")
        else:
            mask = image_to_ndarray(mask_file)

        if not os.path.exists(pred_file):
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"File ${pred_file} does not exist.")
        else:
            pred = image_to_ndarray(pred_file)

        assert mask.shape == pred.shape, "Mask shape and prediction shape should be the same."
        if mask.shape[-1] == 1:
            nbr_class = 2
        else:
            nbr_class = mask.shape[-1]
        return mask.shape[0], mask.shape[1], nbr_class


if __name__ == '__main__':

    img_path = '/home/SPeillet/OCSGE/data/metrics/img'
    # Cas binaire avec du soft
    # mask_path = '/home/SPeillet/OCSGE/data/metrics/pred_soft/binary_case/msk'
    # pred_path = '/home/SPeillet/OCSGE/data/metrics/pred_soft/binary_case/pred'
    # output_path = '/home/SPeillet/OCSGE/'
    # metrics = CLI_Metrics(mask_path, pred_path, output_path, output_type='html', type_classifier='Binary')

    # Cas binaire avec du hard
    # mask_path = '/home/SPeillet/OCSGE/data/metrics/pred_hard/subset_binaire/msk'
    # pred_path = '/home/SPeillet/OCSGE/data/metrics/pred_hard/subset_binaire/pred'
    # output_path = '/home/SPeillet/OCSGE'
    # metrics = CLI_Metrics(mask_path, pred_path, output_path, output_type='html', type_classifier='Binary')

    # Cas multiclass avec du soft
    mask_path = '/home/SPeillet/OCSGE/data/metrics/pred_soft/mcml_case/msk'
    pred_path = '/home/SPeillet/OCSGE/data/metrics/pred_soft/mcml_case/pred'
    output_path = '/home/SPeillet/OCSGE/'
    metrics = CLI_Metrics(mask_path, pred_path, output_path, output_type='json', type_classifier='Multiclass')

    # # # Cas multiclass avec du hard
    # mask_path = '/home/SPeillet/OCSGE/data/metrics/pred_hard/subset_mcml/msk'
    # pred_path = '/home/SPeillet/OCSGE/data/metrics/pred_hard/subset_mcml/pred'
    # output_path = '/home/SPeillet/OCSGE/'
    # metrics = CLI_Metrics(mask_path, pred_path, output_path, output_type='html', type_classifier='Multiclass')

    # Test dataset intégral (cas multiclass en soft)
    # mask_path = '/home/SPeillet/OCSGE/data/metrics/msk'
    # pred_path = '/home/SPeillet/OCSGE/data/metrics/detection_soft/'
    # output_path = '/home/SPeillet/OCSGE/'
    # metrics = CLI_Metrics(mask_path, pred_path, output_path, output_type='html', type_classifier='Multiclass')

    metrics()
