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
from odeon import LOGGER
from odeon.commons.core import BaseTool
from odeon.commons.exception import OdeonError, ErrorCodes
from metrics_factory import Metrics_Factory
# from odeon.commons.metrics import Metrics

ROC_RANGE = np.arange(0, 1.1, 0.1)
BATCH_SIZE = 1
NUM_WORKERS = 1


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
    img_path = '/home/SPeillet/OCSGE/data/metrics/img'
    mask_path = '/home/SPeillet/OCSGE/data/metrics/pred_soft/binary_case/msk'
    pred_path = '/home/SPeillet/OCSGE/data/metrics/pred_soft/binary_case/pred'
    output_path = '/home/SPeillet/OCSGE/data/metrics/pred_soft/binary_case/outputs'
    print('------------------------------------------------------------------------------ ')
    metrics = CLI_Metrics(mask_path, pred_path, output_path, type_classifier='Binary case', threshold=0.5)
    metrics()

