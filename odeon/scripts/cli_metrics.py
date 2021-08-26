"""
Class used as command line interface (CLI) for the class Metrics to analyse the quality of a model's predictions.
Check if the input values of the json configuration file are good and create a dataset as input for the Metrics class.
Then the metrics class will compute metrics, plot confusion matrices (cms) and ROC curves.
This tool handles binary and multi-class cases.
"""
import os
import csv
from odeon import LOGGER
from odeon.commons.core import BaseTool
from odeon.commons.image import image_to_ndarray
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.nn.datasets import MetricsDataset
from odeon.commons.metrics.metrics_factory import Metrics_Factory
from odeon.commons.metrics.metrics import DEFAULTS_VARS


class CLI_Metrics(BaseTool):

    def __init__(self,
                 mask_path,
                 pred_path,
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
        mask_path : str
            Path to the folder containing the masks.
        pred_path : str
            Path to the folder containing the predictions.
        output_path : str
            Path where the report/output data will be created.
        type_classifier : str
            String allowing to know if the classifier is of type binary or multiclass.
        output_type : str, optional
            Desired format for the output file. Could be json, md or html.
            A report will be created if the output type is html or md.
            If the output type is json, all the data will be exported in a dict in order
            to be easily reusable, by default html.
        class_labels : list of str, optional
            Label for each class in the dataset.
            If None the labels of the classes will be of type:  0 and 1 by default None
        weights : list of number, optional
            List of weights to balance the metrics.
            In the binary case the weights are not used in the metrics computation, by default None.
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

        self.type_classifier = type_classifier
        self.threshold = threshold
        self.threshold_range = threshold_range
        self.bit_depth = bit_depth
        self.nb_calibration_bins = nb_calibration_bins
        self.get_normalize = get_normalize

        self.get_metrics_per_patch = get_metrics_per_patch
        self.get_ROC_PR_curves = get_ROC_PR_curves
        self.get_calibration_curves = get_calibration_curves
        self.get_hists_per_metrics = get_hists_per_metrics

        self.mask_files, self.pred_files = self.get_files_from_input_paths()
        self.height, self.width, self.nbr_class = self.get_samples_shapes()

        if class_labels is not None and len(class_labels) != self.nbr_class:
            LOGGER.error('ERROR: parameter labels should have a number of values equal to the number of classes.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input parameter labels is incorrect.")
        else:
            self.class_labels = class_labels

        if weights is not None and len(weights) != self.nbr_class:
            LOGGER.error('ERROR: parameter weigths should have a number of values equal to the number of classes.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input parameter weigths is incorrect.")
        else:
            self.weights = weights

        metrics_dataset = MetricsDataset(self.mask_files,
                                         self.pred_files,
                                         nbr_class=self.nbr_class,
                                         width=self.width,
                                         height=self.height)

        self.metrics = Metrics_Factory(self.type_classifier)(dataset=metrics_dataset,
                                                             output_path=self.output_path,
                                                             type_classifier=self.type_classifier,
                                                             class_labels=self.class_labels,
                                                             output_type=self.output_type,
                                                             weights=self.weights,
                                                             threshold=self.threshold,
                                                             threshold_range=self.threshold_range,
                                                             bit_depth=self.bit_depth,
                                                             nb_calibration_bins=self.nb_calibration_bins,
                                                             get_normalize=get_normalize,
                                                             get_metrics_per_patch=self.get_metrics_per_patch,
                                                             get_ROC_PR_curves=self.get_ROC_PR_curves,
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
        """ WARNING : NOT USED YET
        List all the masks and predicitons files from a csv file.

        Returns
        -------
        List of str
            List of the absolute paths to the masks and predictions files.
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
                LOGGER.warning(f'Problem of matching names between mask {msk} and prediction {pred}.')
        return mask_files, pred_files

    def get_samples_shapes(self):
        """Get the shape of the input masks and predictions.

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
    metrics = CLI_Metrics(mask_path, pred_path, output_path, get_normalize=True, type_classifier='Multiclass')

    # # # Cas multiclass avec du hard
    # mask_path = '/home/SPeillet/OCSGE/data/metrics/pred_hard/subset_mcml/msk'
    # pred_path = '/home/SPeillet/OCSGE/data/metrics/pred_hard/subset_mcml/pred'
    # output_path = '/home/SPeillet/OCSGE/'
    # metrics = CLI_Metrics(mask_path, pred_path, output_path, output_type='html', type_classifier='Multiclass')

    # Test dataset intégral (cas multiclass en soft)
    # mask_path = '/home/SPeillet/OCSGE/data/metrics/msk'
    # pred_path = '/home/SPeillet/OCSGE/data/metrics/detection_soft/'
    # output_path = '/home/SPeillet/OCSGE/'
    # metrics = CLI_Metrics(mask_path, pred_path, output_path, get_normalize=True, type_classifier='Multiclass')

    metrics()
