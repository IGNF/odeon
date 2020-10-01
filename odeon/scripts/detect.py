import os
import pandas as pd
from odeon.commons.core import BaseTool
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.guard import dirs_exist, files_exist, is_valid_dataset_path
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.nn.datasets import PatchDataset

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_generation")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)


class Detector(BaseTool):

    def __init__(self,
                 verbosity,
                 img_size_pixel,
                 resolution,
                 margin,
                 model_name,
                 file_name,
                 batch_size,
                 use_gpu,
                 booster,
                 interruption_recovery,
                 output_path,
                 output_type,
                 export_input,
                 sparse_mode,
                 threshold,
                 dataset=None,
                 zone=None
                 ):
        """

        Parameters
        ----------
        verbosity
        img_size_pixel
        resolution
        margin
        model_name
        file_name
        batch_size
        use_gpu
        booster
        interruption_recovery
        output_path
        output_type
        export_input
        sparse_mode
        threshold
        dataset : str
        zone

        """

        self.verbosity = verbosity
        self.img_size_pixel = img_size_pixel
        self.resolution = resolution
        self.margin = margin
        self.model_name = model_name
        self.file_name = file_name
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.booster = booster
        self.interruption_recovery = interruption_recovery
        self.output_path = output_path
        self.output_type = output_type
        self.export_input = export_input
        self.sparse_mode = sparse_mode
        self.threshold = threshold
        self.df = None

        if zone is not None:

            self.mode = "zone"
            self.zone = zone

        else:

            self.mode = "dataset"
            self.dataset = dataset

        try:

            self.check()
            self.configure()

        except OdeonError as error:

            raise error

        except Exception as error:

            raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                             "something went wrong during detection configuration",
                             stack_trace=error)

    def __call__(self):

        LOGGER.info(self.__dict__)

    def check(self):
        """

        Returns
        -------

        """

        try:

            files_exist([self.file_name])
            dirs_exist([self.output_path, self.dataset])

        except OdeonError as error:

            raise OdeonError(ErrorCodes.ERR_DETECTION_ERROR,
                             "something went wrong during detection configuration",
                             stack_trace=error)

        else:

            pass

    def configure(self):
        """

        Returns
        -------

        """

        if self.mode == "dataset":

            if self.dataset.endswith(".csv"):

                self.df = pd.read_csv(self.dataset, usecols=[0], header=None, names=["img_file"])

            else:

                img_array = [f for f in os.listdir(self.dataset) if os.path.isfile(os.path.join(self.dataset, f))]
                self.df = pd.DataFrame(img_array, columns={"img_file": str})

        else:

            pass
