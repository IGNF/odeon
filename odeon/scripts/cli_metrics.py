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


class CLIMetrics(BaseTool):
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
        self.height, self.width, self.nbr_class = self.get_samples_shapes(mask_bands)

        if self.nbr_class > 2 and self.type_classifier == 'binary':
            LOGGER.error("ERROR: If you have more than 2 classes, please use the classifier type 'multiclass'.")
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input parameter type classifier is incorrect.")

        if self.nbr_class == 2 and class_labels is not None and len(class_labels) == 1:
            if isinstance(class_labels, list):
                self.class_labels = [class_labels[0], 'no_' + class_labels[0]]
            else:
                self.class_labels = ['Positive', 'Negative']
        elif mask_bands is None and class_labels is not None and len(class_labels) != self.nbr_class:
            LOGGER.error('ERROR: parameter labels should have a number of values equal to the number of classes.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input parameter labels is incorrect.")
        elif mask_bands is not None and class_labels is not None and len(class_labels) != len(mask_bands):
            LOGGER.error('ERROR: parameter labels should have a number of input values equal to the number of\
                         selected bands.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR, "The input parameter labels is incorrect.")
        else:
            self.class_labels = class_labels

        if not ((mask_bands is None or pred_bands is None) or (mask_bands is not None and pred_bands is not None)):
            LOGGER.error('ERROR: parameters mask_bands and pred_bands should have the same number of values.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                  "The input parameters mask_bands and pred_bands are incorrect.")
        elif mask_bands is not None and pred_bands is not None:
            if len(mask_bands) != len(pred_bands) or len(mask_bands) > self.nbr_class:
                LOGGER.error('ERROR: parameters mask_bands and pred_bands should have the same number of values.')
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                      "The input parameters mask_bands and pred_bands are incorrect.")
            elif self.type_classifier != 'multiclass':
                LOGGER.error('ERROR: in the fact that even if we are only interested in one band the input set\
                            containing several bands is considered as a multiclass case.')
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                      "The input parameters mask_bands and  or type_classifier should be changed.")
            else:
                # Standardization of band indices with rasterio/gdal, so the user will input the index 1 for the band 0.
                mask_bands, pred_bands = [x - 1 for x in mask_bands], [x - 1 for x in pred_bands]

                # Checks if the bands entered in the configuration file have values corresponding to the bands of the
                # images present in the dataset entered
                self.check_raster_bands(list(range(self.nbr_class)), mask_bands)
                self.check_raster_bands(list(range(self.nbr_class)), pred_bands)

        self.mask_bands = mask_bands
        self.pred_bands = pred_bands

        if weights is not None:
            if self.mask_bands is None and len(weights) != self.nbr_class:
                LOGGER.error('ERROR: parameter weigths should have a number of values equal to the number of classes.')
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                 "The input parameter weigths is incorrect.")
            if self.mask_bands is not None:
                if self.nbr_class > len(self.mask_bands) + 1 and len(weights) != len(self.mask_bands) + 1:
                    LOGGER.error('ERROR: parameter weigths should have a number of values equal to the number of\
                         classes selected.')
                    raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                     "The input parameter weigths is incorrect.")
                elif self.nbr_class <= len(self.mask_bands) + 1 and len(weights) != len(self.mask_bands):
                    LOGGER.error('ERROR: parameter weigths should have a number of values equal to the number of\
                         classes selected.')
                    raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                                     "The input parameter weigths is incorrect.")

        self.weights = weights

        metrics_dataset = MetricsDataset(self.mask_files,
                                         self.pred_files,
                                         nbr_class=self.nbr_class,
                                         width=self.width,
                                         height=self.height,
                                         type_classifier=self.type_classifier)

        self.metrics = MetricsFactory(self.type_classifier)(dataset=metrics_dataset,
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
                                                            get_normalize=get_normalize,
                                                            get_metrics_per_patch=self.get_metrics_per_patch,
                                                            get_ROC_PR_curves=self.get_ROC_PR_curves,
                                                            get_ROC_PR_values=get_ROC_PR_values,
                                                            get_calibration_curves=get_calibration_curves,
                                                            get_hists_per_metrics=get_hists_per_metrics)

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

        for msk, pred in zip(sorted(os.listdir(self.mask_path)), sorted(os.listdir(self.pred_path))):
            if msk == pred:
                mask_files.append(os.path.join(self.mask_path, msk))
                pred_files.append(os.path.join(self.pred_path, pred))
            else:
                LOGGER.warning('Problem of matching names between mask/prediction for %s', msk)
        return mask_files, pred_files

    @staticmethod
    def check_raster_bands(raster_band, proposed_bands):
        """Check if the bands in the configuration file are correct and correspond to the bands in the raster.

        Parameters
        ----------
        raster_band : list
            Bands found by opening the first sample of the dataset.
        proposed_bands : list
            Bands proposed in the configuration file.
        """
        if isinstance(proposed_bands, list) and len(proposed_bands) >= 1:
            if not all((band in raster_band for band in proposed_bands)):
                LOGGER.error('ERROR: the bands in the configuration file do not correspond\
                to the available bands in the image.')
                raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                      "The input parameters mask_bands and pred_bands are incorrect.")
        else:
            LOGGER.error('ERROR: bands must be a list with a length greater than 1.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                  "The input parameters mask_bands and pred_bands are incorrect.")

    def get_samples_shapes(self, mask_bands):
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

        if mask.shape != pred.shape and mask_bands is None:
            LOGGER.error('ERROR: check the dimensions of the inputs masks and detections. \
                Those input data should have the same dimensions.')
            LOGGER.info('INFO: The parameters mask_bands and pred_bands could be used to \
            do band selection in order to use apply the tool to data with different channel numbers.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "Detections and masks have different dimensions.")

        if mask.shape[-1] == 1:
            nbr_class = 2
        else:
            nbr_class = min(mask.shape[-1], pred.shape[-1])

        assert len(np.unique(mask.flatten())) <= nbr_class, \
            "Mask must contain a maximum number of unique values equal to the number of classes"

        return mask.shape[0], mask.shape[1], nbr_class
