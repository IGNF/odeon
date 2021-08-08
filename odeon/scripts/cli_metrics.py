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
import torch
import rasterio
from odeon import LOGGER
from odeon.commons.core import BaseTool
from odeon.commons.exception import OdeonError, ErrorCodes
# from odeon.commons.metrics import Metrics

BATCH_SIZE = 1
NUM_WORKERS = 1

class Metrics():
    pass


class CLI_Metrics(BaseTool):

    def __init__(self,
                 input_path,
                 output_path,
                 classes=None,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):

        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.masks_files, self.pred_files = self.files_from_input_path()
        
        self.metrics = Metrics(mask_files=self.masks_files,
                               pred_files=self.pred_files,
                               classes=self.classes)

    def __call__(self):
        self.metrics()

    def files_from_input_path(self):
        """[summary]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        OdeonError
            [description]
        """
        if not os.path.exists(self.input_path):
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"file/folder ${self.input_path} does not exist.")
        else:
            if os.path.splitext(self.input_path)[1] == '.csv':
                mask_files, pred_files = self.read_csv_sample_file()
            elif os.path.isdir(self.input_path):
                mask_files, pred_files = self.list_files_from_dir()
            else:
                LOGGER.error('ERROR: the input path shoud point to a csv file or to a dataset directories.')
        return mask_files, pred_files

    def read_csv_sample_file(self):
        """Read a sample CSV file and return a list of image files and a list of mask files.
        CSV file should contain image pathes in the first column and mask pathes in the second.

        Parameters
        ----------
        input_path : str
            path to sample CSV file

        Returns
        -------
        Tuple[list, list]
            A list of image pathes and a list of mask pathes.
        """
        mask_files = []
        pred_files = []

        with open(self.input_path) as csvfile:
            sample_reader = csv.reader(csvfile)
            for item in sample_reader:
                mask_files.append(item['msk_file'])
                pred_files.append(item['img_output_file'])
        return mask_files, pred_files

    def list_files_from_dir(self):
        """List files in a diretory and return a list of image files and a list of mask files.
        Dataset directory should contain and 'msk' folder and a 'pred' folder.
        Masks and predictions should have the same names.

        Parameters
        ----------
        input_path : str
            path to the folders with the masks and the predictions.

        Returns
        -------
        Tuple[list, list]
            a list of image pathes and a list of mask pathes
        """
        path_msk = os.path.join(self.input_path, 'msk')
        path_pred = os.path.join(self.input_path, 'pred')

        mask_files, pred_files = [], []

        for msk, pred in zip(sorted(os.listdir(path_msk)), sorted(os.listdir(path_pred))):
            if msk == pred:
                mask_files.append(os.path.join(path_msk, msk))
                pred_files.append(os.path.join(path_pred, pred))

            else:
                LOGGER.warning(f'Problem of matching names between mask {msk} and prediction {pred}.')

        return mask_files, pred_files
