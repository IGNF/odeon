"""
Stats object which allows to make the link between the CLI and the statistics object of the odeon.commons module.
The commons/Statistics object then will compute descriptive statistics on a dataset on :
- the bands of the images
- the classes present in the masks
- the globality of the dataset (cf. Statistics documentation)
The input parameters pass to the Stats object will come from a configuration file.
"""

import os
import csv
import torch
import rasterio
from odeon import LOGGER
from odeon.commons.core import BaseTool
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.statistics import Statistics
from odeon.nn.transforms import Compose, Rotation90, Rotation, Radiometry, ToDoubleTensor
from odeon.nn.datasets import PatchDataset

BATCH_SIZE = 1
NUM_WORKERS = 1
BIT_DEPTH = '8 bits'
NBR_BINS = 15
GET_SKEWNESS_KURTOSIS = False


class Stats(BaseTool):
    """Class Stats to create dataset and use the commons.statistics module.
    """
    def __init__(self,
                 input_path,
                 output_path,
                 output_type=None,
                 image_bands=None,
                 mask_bands=None,
                 data_augmentation=None,
                 bins=None,
                 nbr_bins=NBR_BINS,
                 get_skewness_kurtosis=GET_SKEWNESS_KURTOSIS,
                 bit_depth=BIT_DEPTH,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):

        """Init function of Stats class.

        Parameters
        ----------
        input_path : str
            Path to .csv file describing the input dataset or a directory where the images and masks are stored.
        output_path: str
            Path where the report with the computed statistics will be created.
        image_bands: list
            List of the selected bands in the dataset images bands.
        mask_bands: list
            List of the selected bands in the dataset masks bands. (Selection of the classes)
        data_augmentation: list/str
            Data augmentation to apply in the input dataset.
        bins: list
            List of the bins to build the histograms of the image bands.
        nbr_bins: int.
            If bins is not given in input, the list of bins will be created with the nbr_bins defined here.
        get_skewness_kurtosis: bool
            Boolean to compute or not skewness and kurtosis.
        bit_depth: str
            The number of bits used to represent each pixel in an image.
        batch_size: int
            The number of image in a batch.
        num_workers: int
            Number of workers to use in the pytorch dataloader.
        """
        self.input_path = input_path

        if not os.path.exists(output_path):
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Output folder ${output_path} does not exist.")
        elif not os.path.isdir(output_path):
            raise OdeonError(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"Output path ${output_path} should be a folder.")
        else:
            self.output_path = output_path

        if output_type in ['md', 'json', 'html', 'terminal']:
            self.output_type = output_type
        else:
            LOGGER.error('ERROR: the output file can only be in md, json, html or directly displayed on the terminal.')
            self.output_type = 'html'

        self.bins = bins
        self.nbr_bins = nbr_bins
        self.bit_depth = bit_depth
        self.get_skewness_kurtosis = get_skewness_kurtosis
        # self.device = self.check_device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers

        if not os.path.exists(self.input_path):
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"file ${self.input_path} does not exist.")
        else:
            if os.path.splitext(self.input_path)[1] == '.csv':
                self.image_files, self.mask_files = self.read_csv_sample_file(self.input_path)
            elif os.path.isdir(self.input_path):
                self.image_files, self.mask_files = self.list_files_from_dir(self.input_path)
            else:
                LOGGER.error('ERROR: the input path shoud point to a csv file or to a dataset directories.')

        # Bands obtained by opening the first sample of the dataset
        read_img_bands, self.img_heigth, self.img_width = self.get_raster_info(self.image_files[0])
        read_msk_bands, self.msk_heigth, self.msk_width = self.get_raster_info(self.mask_files[0])

        if image_bands is not None:
            self.check_raster_bands(read_img_bands, image_bands)
        else:
            image_bands = read_img_bands

        if mask_bands is not None:
            self.check_raster_bands(read_msk_bands, mask_bands)
        else:
            mask_bands = read_msk_bands

        self.image_bands, self.mask_bands = image_bands, mask_bands

        if self.img_heigth != self.msk_heigth or self.img_width != self.msk_width:
            LOGGER.warning(f"""WARNING: images and masks dimensions are not the same.
                                        images: {self.img_heigth} x {self.img_width}
                                        masks: {self.msk_heigth} x {self.msk_width}""")

        # Data augmentation
        self.transform = None
        if data_augmentation is not None:
            transformation_dict = {
                "rotation90": Rotation90(),
                "rotation": Rotation(),
                "radiometry": Radiometry()
            }
            transformation_conf = data_augmentation
            transformation_keys = transformation_conf \
                if isinstance(transformation_conf, list) else [transformation_conf]

            self.transformation_functions = list({
                value for key, value in transformation_dict.items()
                if key in transformation_keys
            })
            self.transformation_functions.append(ToDoubleTensor())
            self.transform = Compose(self.transformation_functions)

        self.dataset = PatchDataset(self.image_files,
                                    self.mask_files,
                                    transform=self.transform,
                                    width=min(self.img_width, self.msk_width),
                                    height=min(self.img_heigth, self.msk_heigth),
                                    image_bands=self.image_bands,
                                    mask_bands=self.mask_bands)

        self.statistics = Statistics(dataset=self.dataset,
                                     output_path=self.output_path,
                                     output_type=output_type,
                                     get_skewness_kurtosis=self.get_skewness_kurtosis,
                                     bit_depth=self.bit_depth,
                                     bins=self.bins,
                                     nbr_bins=self.nbr_bins,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers)

    def __call__(self):
        """
        Function to generate an output file when the instance is called.
        """
        self.statistics()

    def read_csv_sample_file(self, input_path):
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
        image_files = []
        mask_files = []

        with open(input_path) as csvfile:
            sample_reader = csv.reader(csvfile)
            for item in sample_reader:
                image_files.append(item[0])
                mask_files.append(item[1])
        return image_files, mask_files

    def list_files_from_dir(self, input_path):
        """List files in a diretory and return a list of image files and a list of mask files.
        Dataset directory should contain and 'img' folder and a 'msk' folder.
        Images and masks should have the same names.

        Parameters
        ----------
        input_path : str
            Path to the dataset directory.

        Returns
        -------
        Tuple[list, list]
            A list of image pathes and a list of mask pathes.
        """

        path_img = os.path.join(input_path, 'img')
        path_msk = os.path.join(input_path, 'msk')

        image_files, mask_files = [], []

        for img, msk in zip(sorted(os.listdir(path_img)), sorted(os.listdir(path_msk))):
            if img == msk:
                image_files.append(os.path.join(path_img, img))
                mask_files.append(os.path.join(path_msk, msk))

            else:
                LOGGER.warning(f'Problem of matching names between image {img} and mask {msk}.')

        return image_files, mask_files

    def get_raster_info(self, path_raster):
        """Give the number of bands in a raster.

        Parameters
        ----------
        path_raster : str
            Path to the raster.

        Returns
        -------
        int
            Number of bands in the raster.
        """
        with rasterio.open(path_raster, 'r') as raster:
            return list(range(raster.count)), raster.height, raster.width

    def check_raster_bands(self, raster_band, proposed_bands):
        """Check if the bands in the configuration file are correct and correspond to the bands in the raster.

        Parameters
        ----------
        raster_band : list
            Bands found by opening the first sample of the dataset.
        proposed_bands : list
            Bands proposed in the configuration file.
        """
        if isinstance(proposed_bands, list) and len(proposed_bands) > 1:
            if not all([band in raster_band for band in proposed_bands]):
                LOGGER.error(f'ERROR: the bands in the configuration file do not correspond\
                to the available bands in the image. The bands in the image are : {raster_band}.')
        else:
            LOGGER.error('ERROR: bands must be a list with a length greater than 1.')

    def check_device(self, proposed_device):
        """ Check if the device pass in the configuration file in available.
        If not, use of the cpu.

        Parameters
        ----------
        proposed_device: int/list
            Device(s) in the configuration file to use for the Stats tool.
        """
        default_device = 'cpu'
        # check if device as the good format
        if proposed_device == 'cpu':
            pass
        elif proposed_device.startswith('cuda:'):
            id_device = proposed_device.split(':')[1]
            cuda_available = torch.cuda.is_available()
            devices_available = list(range(torch.cuda.device_count()))

            if cuda_available and id_device in devices_available:
                # If verbosity
                LOGGER.info(f'INFO: device used : {default_device}')
                LOGGER.info(f"""GPU: {torch.cuda.get_device_name(id_device)}
                Memory Usage:
                Allocated:, {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB
                Cached:   , {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB""")
                return proposed_device

            else:
                LOGGER.warning(f'WARNING: The device {proposed_device} is not available.\
                    The gpus available are: {devices_available}.')

        else:
            LOGGER.warning("WARNING: device should be of the form 'cpu' or \
                'cuda: X' with X the id of the selected GPU.")

        LOGGER.info(f'INFO: device used : {default_device}')
        return default_device


if __name__ == '__main__':
    input_path = "/home/SPeillet/OCSGE/outputs/generation/train"
    output_path = "/home/SPeillet/OCSGE/"
    stats = Stats(input_path, output_path, output_type='md', get_skewness_kurtosis=True)
    stats()
