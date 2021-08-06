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

class CLI_Metrics(BaseTool):

    def __init__(self,
                 input_path,
                 output_path,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):
        
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.metrics = Metrics(input_path=input_path,
        )

    def __call__(self):
        #self.metrics()
