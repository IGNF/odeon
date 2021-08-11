import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import itertools
from odeon import LOGGER
from odeon.commons.image import image_to_ndarray
from odeon.commons.exception import OdeonError, ErrorCodes
from abc import ABC, abstractmethod
from metrics import Metrics
# from odeon.commons.metrics import Metrics

ROC_RANGE = np.arange(0, 1.1, 0.1)
BATCH_SIZE = 1
NUM_WORKERS = 1


class MC_ML_Metrics(Metrics):
    pass