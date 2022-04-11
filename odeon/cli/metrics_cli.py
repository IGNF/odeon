"""
Class used as command line interface (CLI) for the class Metrics to analyse the quality of a model's predictions.
Check if the input values of the json configuration file are good and create a dataset as input for the Metrics class.
Then the metrics class will compute metrics, plot confusion matrices (cms) and ROC curves.
This tool handles binary and multi-class cases.
"""
import os
from datetime import datetime
import numpy as np
import rasterio
from odeon.commons.core import BaseTool
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon import LOGGER
from odeon.nn.datasets import MetricsDataset
from odeon.commons.metric.metrics_factory import MetricsFactory
from odeon.commons.metric.metrics import DEFAULTS_VARS
from odeon.commons.guard import check_raster_bands


class MetricsCLI(BaseTool):
    """
    Class to check variables coming from the CLI and create an instance of the class Metrics.
    """
    def __init__(self,
                 mask_path,
                 pred_path,
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
        mask_path : str
            Path to the folder containing the masks.
        pred_path : str
            Path to the folder containing the predictions.
        output_path : str
            Path where the report/output data will be created.
        type_classifier : str
            String allowing to know if the classifier is of type binary or multiclass.
        in_prob_range : boolean,
            Boolean to be set to true if the values in the predictions passed as inputs are between 0 and 1.
            If not, set the parameter to false so that the tool modifies the values to be normalized between 0 and 1.
        output_type : str, optional
            Desired format for the output file. Could be json, md or html.
            A report will be created if the output type is html or md.
            If the output type is json, all the data will be exported in a dict in order
            to be easily reusable, by default html.
        class_labels : list of str, optional
            Label for each class in the dataset, by default None.
        mask_bands: list of int
            List of the selected bands in the dataset masks bands. (Selection of the classes)
        pred_bands: list of int
            List of the selected bands in the dataset preds bands. (Selection of the classes)
        weights : list of number, optional
            List of weights to balance the metrics.
            In the binary case the weights are not used in the metrics computation, by default None.
        threshold : float, optional
            Value between 0 and 1 that will be used as threshold to binarize data if they are soft.
            Use for macro, micro cms and metrics for all strategies, by default 0.5.
        n_thresholds : int, optional
            Number of thresholds used in the computation of ROC and PR, by default 10.
        bit_depth : str, optional
            The number of bits used to represent each pixel in a mask/prediction, by default '8 bits'
        bins: list of float, optional
            List of bins used for the creation of histograms.
        n_bins : int, optional
            Number of bins used in the construction of calibration curves, by default 10.
        get_normalize : bool, optional
            Boolean to know if the user wants to generate confusion matrices with normalized values, by default True
        get_metrics_per_patch : bool, optional
            Boolean to know if the user wants to compute metrics per patch and export them in a csv file.
            Metrics will be also computed if the parameter get_hists_per_metrics is True but a csv file
            won't be created, by default True
        get_ROC_PR_curves : bool, optional
            Boolean to know if the user wants to generate ROC and PR curves, by default True
        get_ROC_PR_values: bool, optional
            Boolean to know if the user wants a csv file with values used to generate ROC/PR curves, by default False
        get_calibration_curves : bool, optional
            Boolean to know if the user wants to generate calibration curves, by default True
        get_hists_per_metrics : bool, optional
            Boolean to know if the user wants to generate histogram for each metric.
            Histograms created using the parameter threshold, by default True.
        """
        self.mask_path = mask_path
        self.pred_path = pred_path

        if os.path.exists(output_path) and os.path.isdir(output_path):
            name_output_path = os.path.join(output_path,
                                            'metrics_report_' + datetime.today().strftime("%Y_%m_%d_%H_%M_%S"))
            os.makedirs(name_output_path)
            self.output_path = name_output_path
        else:
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Output folder ${output_path} does not exist.")

        if output_type in ['md', 'json', 'html']:
            self.output_type = output_type
        else:
            LOGGER.error('ERROR: the output file can only be in md, json, html.')

        self.in_prob_range = in_prob_range
        self.type_classifier = type_classifier.lower()
        self.threshold = threshold
        self.n_thresholds = n_thresholds
        self.bit_depth = bit_depth
        self.n_bins = n_bins
        self.bins = bins
        self.get_normalize = get_normalize
        self.get_metrics_per_patch = get_metrics_per_patch
        self.get_ROC_PR_curves = get_ROC_PR_curves
        self.get_ROC_PR_values = get_ROC_PR_values
        self.get_calibration_curves = get_calibration_curves
        self.get_hists_per_metrics = get_hists_per_metrics
        self.mask_files, self.pred_files = self.get_files_from_input_paths()
        self.height, self.width, mask_class, pred_class = self.get_samples_shapes()

        # Check mask_bands/pred_bands parameters
        if mask_bands is not None:
            if pred_bands is None:
                pred_bands = mask_bands
            # Standardization of band indices with rasterio/gdal, so the user will input the index 1 for the band 0.
            mask_bands, pred_bands = [x - 1 for x in mask_bands], [x - 1 for x in pred_bands]
            # Checks if the bands entered in the configuration file have values corresponding to the bands of the
            # images present in the dataset entered
            check_raster_bands(np.arange(mask_class), mask_bands)
            check_raster_bands(np.arange(pred_class), pred_bands)

            if len(mask_bands) == len(pred_bands):
                if self.type_classifier == 'binary' and len(mask_bands) > 1:
                    LOGGER.error('ERROR: bands must be a list with a length greater than 1.')
                    raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                     "The input parameters mask_bands and pred_bands are incorrect.")
                self.mask_bands = mask_bands
                self.pred_bands = pred_bands
                self.nbr_class = len(mask_bands) if self.type_classifier == 'multiclass' else 2
            else:
                LOGGER.error('ERROR: parameters mask_bands and pred_bands should have the same number of values if\
                             pred_bands is defined.')
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                 "The input parameters mask_bands and pred_bands are incorrect.")
        else:
            self.mask_bands = self.pred_bands = None
            if min(mask_class, pred_class) > 2 and self.type_classifier == 'binary':
                LOGGER.error("ERROR: If you have more than 2 classes, please use the classifier type 'multiclass' or\
                             select a band with the parameters mask_bands/pred_bands")
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                 "The input parameter type classifier is incorrect.")
            elif self.type_classifier == 'binary':
                self.nbr_class = 2
            else:
                self.nbr_class = min(mask_class, pred_class)

        # Check labels parameter
        if class_labels is not None:
            if self.nbr_class == 2 and len(class_labels) == 1:
                self.class_labels = [class_labels[0], 'no_' + class_labels[0]]
            elif len(class_labels) == self.nbr_class:
                self.class_labels = class_labels
            else:
                LOGGER.error('ERROR: parameter labels should have a number of values equal to the number of classes.')
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                 "The input parameter labels is incorrect.")
        else:
            if self.nbr_class == 2:
                self.class_labels = ['Positive', 'Negative']
            else:
                self.class_labels = [f'class {i + 1}' for i in range(self.nbr_class)]

        # Check weights parameter
        if weights is not None:
            if (len(weights) == self.nbr_class) or \
               (len(self.mask_bands) + 1 < min(mask_class, pred_class) and len(weights) == self.nbr_class + 1):
                # Gives user the possibility to fix the weight for the 'Other' class when this one is created.
                self.weights = np.array(weights)
            else:
                LOGGER.error('ERROR: parameter weigths should have a number of values equal to the number of classes.')
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                 "The input parameter weigths is incorrect.")
        else:
            self.weights = np.ones(self.nbr_class) if self.type_classifier == 'multiclass' else None

        # For class selection, if selected bands + 1 < min(mask_bands, pred_bands) then 'Other' class is created
        if self.mask_bands is not None and len(self.mask_bands) + 1 < min(mask_class, pred_class) \
           and self.type_classifier == 'multiclass':
            # Add 1 because we create a class other for all the bands not selected.
            self.nbr_class += 1
            self.class_labels.append('Other')
            # If the user doesn't define a weight for the Other class, this one will have a weight of 0.
            if len(self.weights) != self.nbr_class:
                self.weights = np.append(self.weights, 0.0)

        self.metrics_dataset = MetricsDataset(self.mask_files,
                                              self.pred_files,
                                              nbr_class=self.nbr_class,
                                              type_classifier=self.type_classifier,
                                              mask_bands=self.mask_bands,
                                              pred_bands=self.pred_bands,
                                              width=self.width,
                                              height=self.height)

        self.metrics = MetricsFactory(self.type_classifier)(dataset=self.metrics_dataset,
                                                            output_path=self.output_path,
                                                            type_classifier=self.type_classifier,
                                                            in_prob_range=self.in_prob_range,
                                                            class_labels=self.class_labels,
                                                            output_type=self.output_type,
                                                            mask_bands=self.mask_bands,
                                                            pred_bands=self.pred_bands,
                                                            weights=self.weights,
                                                            threshold=self.threshold,
                                                            n_thresholds=self.n_thresholds,
                                                            bit_depth=self.bit_depth,
                                                            bins=self.bins,
                                                            n_bins=self.n_bins,
                                                            get_normalize=self.get_normalize,
                                                            get_metrics_per_patch=self.get_metrics_per_patch,
                                                            get_ROC_PR_curves=self.get_ROC_PR_curves,
                                                            get_ROC_PR_values=self.get_ROC_PR_values,
                                                            get_calibration_curves=self.get_calibration_curves,
                                                            get_hists_per_metrics=self.get_hists_per_metrics)

    def __call__(self):
        """
        Call the metrics object. Ouputs files are created when the object is called.
        """
        self.metrics()

    def get_files_from_input_paths(self):
        """
        Check if the inputs folders exits and list all the files from the mask and prediction input folders.

        Returns
        -------
        List of str
            List of the absolute paths to the masks and predictions files.

        Raises
        ------
        OdeonError
            Mask folder does not exist.
        OdeonError
            Prediction folder does not exist.
        """
        if not os.path.exists(self.mask_path):
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Masks folder {self.mask_path} does not exist.")
        elif not os.path.exists(self.pred_path):
            LOGGER.error('ERROR: Predictions folder %s does not exist.', self.pred_path)
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Predictions folder {self.pred_path} does not exist.")
        else:
            if os.path.isdir(self.mask_path) and os.path.isdir(self.pred_path):
                mask_files, pred_files = self.list_files_from_dir()
            else:
                LOGGER.error('ERROR: the input paths shoud point to dataset directories.')
        return mask_files, pred_files

    def list_files_from_dir(self):
        """ List all the files from the mask and prediction input folders.

        Returns
        -------
        List of str
            List of the absolute paths to the masks and predictions files.
        """
        mask_files, pred_files = [], []

        for name_file in os.listdir(self.mask_path):
            file_msk = os.path.join(self.mask_path, name_file)
            file_pred = os.path.join(self.pred_path, name_file)
            if os.path.exists(file_msk) and os.path.exists(file_pred):
                mask_files.append(file_msk)
                pred_files.append(file_pred)
            else:
                LOGGER.warning('Problem of matching names between mask/prediction for %s', msk)
        return mask_files, pred_files

    def get_samples_shapes(self):
        """Get the shape of the input masks and predictions.

        Parameters
        ----------
        mask_bands : list
            Band indices that the user wants to select.

        Returns
        -------
        Tuple of int
            Height, width and number of bands of the input masks and predictions.

        Raises
        ------
        OdeonError
            Mask folder does not exist.
        OdeonError
            Prediction folder does not exist.
        """
        mask_file, pred_file = self.mask_files[0], self.pred_files[0]

        if os.path.exists(mask_file):
            with rasterio.open(mask_file) as mask_raster:
                mask = mask_raster.read().swapaxes(0, 2).swapaxes(0, 1)
        else:
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"File ${mask_file} does not exist.")

        if os.path.exists(pred_file):
            with rasterio.open(pred_file) as pred_raster:
                pred = pred_raster.read().swapaxes(0, 2).swapaxes(0, 1)
        else:
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"File ${pred_file} does not exist.")

        if mask.shape[0:-1] != pred.shape[0:-1]:
            LOGGER.error('ERROR: check the width/height of the inputs masks and detections. \
                Those input data should have the same width/height.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "Detections and masks have different width/height.")

        assert len(np.unique(mask.flatten())) <= mask.shape[-1], \
            "Mask must contain a maximum number of unique values equal to the number of classes"

        return mask.shape[0], mask.shape[1], mask.shape[-1], pred.shape[-1]
