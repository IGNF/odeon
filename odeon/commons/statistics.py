"""
Statistics class to compute descriptive statistics on a dataset.

    Statistics are computed on :
        * the bands of the images: min, max, mean, std and the total histograms
            for each band. (Skewness and kurtosis are optional).
        * the classes present in the masks:
            - regu L1: Class-Balanced Loss Based on Effective Number of
              Samples 1/frequency(i)
            - regu L2: Class-Balanced Loss Based on Effective Number of
              Samples 1/sqrt(frequency(i))
            - pixel freq: Overall share of pixels labeled with a given class.
            - freq 5%pixel: Share of samples with at least
                5% pixels of a given class.
                The lesser, the more concentrated on a few samples a class is.
            - auc: Area under the Lorenz curve of the pixel distribution of a
                given class across samples. The lesser, the more concentrated
                on a few samples a class is. Equals pixel freq if the class is
                the samples are either full of or empty from the class. Equals
                1 if the class is homogeneously distributed across samples.
        * the globality of the dataset: (either with all classes or without the
          last class if we are not in the binary case)
            - Percentage of pixels shared by several classes (share multilabel)
            - the number of classes in an image (avg nb class in patch)
            - the average entropy (avg entropy)

    As output, the instance of this class can either generate a JSON file
    containing the computed statistics or display directly in console the
    obtained results.
"""

import os
from odeon import LOGGER
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from torch.utils.data import DataLoader
from tqdm import tqdm
from odeon.commons.reports.report_factory import Report_Factory
from odeon.commons.exception import OdeonError, ErrorCodes

BATCH_SIZE = 1
NUM_WORKERS = 1
BIT_DEPTH = '8 bits'
GET_SKEWNESS_KURTOSIS = False
NBR_BINS = 15


class Statistics():

    def __init__(self,
                 dataset,
                 output_path,
                 output_type=None,
                 get_skewness_kurtosis=GET_SKEWNESS_KURTOSIS,
                 bit_depth=BIT_DEPTH,
                 bins=None,
                 nbr_bins=NBR_BINS,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):
        """
        Init function of Statistics class.

        Parameters
        ----------
        dataset :  PatchDataset
            Dataset from odeon.nn.datasets which contains the images and masks.
        output_path: str
            Path where the report with the computed statistics will be created.
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
        # Input arguments
        self.dataset = dataset

        if not os.path.exists(output_path):
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"Output folder ${output_path} does not exist.")
        else:
            self.output_path = output_path

        if output_type in ['md', 'json', 'html', 'terminal']:
            self.output_type = output_type
        else:
            LOGGER.error('ERROR: the output file can only be in md, json, html or directly displayed on the terminal.')
            self.output_type = 'html'

        self.nbr_bands = len(self.dataset.image_bands)
        self.nbr_classes = len(self.dataset.mask_bands)
        self.nbr_pixels_per_patch = self.dataset.height * self.dataset.width
        self.nbr_total_pixel = len(self.dataset) * self.nbr_pixels_per_patch

        self.depth_dict = {'keep':  1,
                           '8 bits': 255,
                           '12 bits': 4095,
                           '14 bits': 16383,
                           '16 bits': 65535}

        if bit_depth in self.depth_dict.keys():
            self.bit_depth = bit_depth
        else:
            self.bit_depth = BIT_DEPTH
            LOGGER.warning(f"""WARNING: the pixel depth input in the configuration file is not correct.
                            For the rest of the computations we will consider that the images in your
                            input dataset are in {BIT_DEPTH}.""")
        self.zeros_pixels = 0
        self.nbr_bins = nbr_bins
        self.bins = self.get_bins(bins)
        self.get_skewness_kurtosis = get_skewness_kurtosis

        # To compute the stats for the images bands.
        self.min = [float('inf')] * self.nbr_bands
        self.max = [-float('inf')] * self.nbr_bands
        self._sum = self._sumSq = self.means = self.std = np.zeros(self.nbr_bands)  # Vars to compute the mean and std.
        if self.get_skewness_kurtosis:
            self.skewness = np.zeros(self.nbr_bands)
            self.kurtosis = np.zeros(self.nbr_bands)

        self.batch_size = batch_size
        self.num_workers = num_workers
        assert self.batch_size <= len(self.dataset), "batch_size must be lower than the length of the dataset"

        if len(self.dataset) % self.batch_size == 0:
            self.nbr_batches = len(self.dataset)//self.batch_size
        else:
            self.nbr_batches = len(self.dataset)//self.batch_size + 1

        # Dataframes creation
        self.df_dataset, self.df_bands_stats, self.df_classes_stats, self.df_global_stats, self.bands_hists =\
            self.create_data_for_stats()

        self.scan_dataset()
        self.compute_stats()
        self.report = Report_Factory(self)

    def __call__(self):
        """
        Function to display or to generate an output file when the instance is called.
        """
        self.report.create_report()

    def get_bins(self, bins):
        """Transforms the bins passed in input to normalize values to be used during the scan of the data,
        (which are normalized in PatchDataset). If bins are not defined, they will be created thanks to the attribut
        nbr_bins.

        Parameters
        ----------
        bins : list/None
            Bins to compute the histogram of the image bands.

        Returns
        -------
        Tuple(list, list)
            bins: Bins in the range of the original images pixel values.
            bins_norms: bins with normalized values.
        """
        max_pixel_value = self.depth_dict[self.bit_depth]
        if bins is None:
            bins = [round((i/self.nbr_bins) * max_pixel_value, 3) for i in range(self.nbr_bins)]
            bins.append(max_pixel_value)
        return bins

    def create_data_for_stats(self):
        """Create dataframes and list to store the computed statistics.

        Returns
        -------
        list(pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list)
            Dataframes with the right dimensions and headers and the list for the histograms.
        """
        # Creation of the dataframe for the global stats
        # If we are in the multiclass case, we calculate the stats also without the last class.
        if self.nbr_classes > 2:
            df_global_stats = pd.DataFrame(index=['all classes', 'without last class'],
                                           columns=['share multilabel', 'avg nb class in patch', 'avg entropy'])
            df_global_stats.loc['without last class', 'share multilabel'] = 0
        else:  # If we are in a binary case
            df_global_stats = pd.DataFrame(index=['all classes'],
                                           columns=['share multilabel', 'avg nb class in patch', 'avg entropy'])
        df_global_stats.loc['all classes', 'share multilabel'] = 0

        header_list = [f'class {i}' for i in range(1, self.nbr_classes+1)]
        df_dataset = pd.DataFrame(index=range(len(self.dataset)), columns=header_list)

        if self.get_skewness_kurtosis:
            header_bands = ['min', 'max', 'mean', 'std', 'skewness', 'kurtosis']
        else:
            header_bands = ['min', 'max', 'mean', 'std']
        df_bands_stats = pd.DataFrame(index=[f'band {i}' for i in range(1, self.nbr_bands+1)], columns=header_bands)

        df_classes_stats = pd.DataFrame(index=[f'class {i}' for i in range(1, self.nbr_classes+1)],
                                        columns=['regu L1', 'regu L2', 'pixel freq', 'freq 5% pixel', 'auc'])

        bands_hists = [np.zeros(len(self.bins) - 1) for _ in range(self.nbr_bands)]

        return df_dataset, df_bands_stats, df_classes_stats, df_global_stats, bands_hists

    def scan_dataset(self):
        """
        Iterate over the dataset in one pass, collect all statistics on images and classes
        and compute directly global statistics.
        """
        nb_class_in_patch, list_entropy = [], []
        if self.nbr_classes > 2:  # If nbr classes > 2, we compute stats without last class.
            nb_class_in_patch_wlc, list_entropy_wlc = [], []

        # Pass over the data to collect stats, hist, sum and counts.
        stat_dataloader = DataLoader(self.dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)

        index = 0
        for sample in tqdm(stat_dataloader, desc='First pass', leave=True):
            for image, mask in zip(sample['image'], sample['mask']):
                image = self.to_pixel_input_range(image.numpy().swapaxes(0, 2).swapaxes(0, 1))
                self.zeros_pixels += np.count_nonzero(np.sum(image, axis=2) == 0)
                mask = mask.numpy().swapaxes(0, 2).swapaxes(0, 1)
                for idx_band in range(self.nbr_bands):
                    vect_band = image[:, :, idx_band].flatten()

                    cr_min = np.min(vect_band)
                    if cr_min < self.min[idx_band]:
                        self.min[idx_band] = cr_min

                    cr_max = np.max(vect_band)
                    if cr_max > self.max[idx_band]:
                        self.max[idx_band] = cr_max

                    self._sum[idx_band] += np.sum(vect_band)

                    # Cumulative addition by band of the histograms of each image.
                    current_bins_counts = np.histogram(vect_band, self.bins)[0]
                    self.bands_hists[idx_band] = np.add(self.bands_hists[idx_band], current_bins_counts)

                for idx_class in range(1, self.nbr_classes+1):
                    vect_class = mask[:, :, idx_class-1].flatten()
                    self.df_dataset.loc[index, f'class {idx_class}'] = np.count_nonzero(vect_class)
                index += 1

                # Information storage for statistics.
                self.df_global_stats.loc['all classes', 'share multilabel'] += \
                    np.count_nonzero(np.sum(mask, axis=2) > 1)
                # Sum of each band to get the total pixels present per class in a mask.
                vect_sum_class = np.sum(mask, axis=(0, 1))
                nb_class_in_patch.append(np.count_nonzero(vect_sum_class))

                if all(np.equal(vect_sum_class, np.zeros(self.nbr_classes))):
                    sample_entropy = 0  # A vector of 0 passing to the entropy function returns Nans vector.
                else:
                    vect_normalize = vect_sum_class/self.nbr_pixels_per_patch
                    sample_entropy = entropy(vect_normalize)
                list_entropy.append(sample_entropy)

                if self.nbr_classes > 2:
                    self.df_global_stats.loc['without last class', 'share multilabel'] += \
                         np.count_nonzero(np.sum(mask[:, :, :self.nbr_classes-1], axis=2) > 1)
                    nb_class_in_patch_wlc.append(np.count_nonzero(vect_sum_class[:-1]))

                    if all(np.equal(vect_sum_class[:-1], np.zeros(self.nbr_classes-1))):
                        sample_entropy_wlc = 0
                    else:
                        vect_normalize_wlc = vect_sum_class[:-1]/self.nbr_pixels_per_patch
                        sample_entropy_wlc = entropy(vect_normalize_wlc)
                    list_entropy_wlc.append(sample_entropy_wlc)

        self.means = self._sum / self.nbr_total_pixel

        # Second pass to compute variance:
        for sample in tqdm(stat_dataloader, desc='Second pass', leave=True):
            for image, _ in zip(sample['image'], sample['mask']):
                image = self.to_pixel_input_range(image.numpy().swapaxes(0, 2).swapaxes(0, 1))
                for idx_band in range(self.nbr_bands):
                    vect_band = image[:, :, idx_band].flatten()
                    self._sumSq[idx_band] += np.sum(np.square(vect_band - self.means[idx_band]))

        self.std = np.sqrt((self._sumSq / (self.nbr_total_pixel - 1)))

        # Third pass to compute sknewness and kurtosis
        if self.get_skewness_kurtosis:
            for sample in tqdm(stat_dataloader, desc='Third pass', leave=True):
                for image, _ in zip(sample['image'], sample['mask']):
                    image = self.to_pixel_input_range(image.numpy().swapaxes(0, 2).swapaxes(0, 1))
                    for idx_band in range(self.nbr_bands):
                        vect_band = image[:, :, idx_band].flatten()
                        vect_band_std = (vect_band - self.means[idx_band]) / self.std[idx_band]
                        self.skewness[idx_band] += np.sum(np.power(vect_band_std, 3))
                        self.kurtosis[idx_band] += np.sum(np.power(vect_band_std, 4))

        self.df_global_stats.loc['all classes', 'share multilabel'] /= self.nbr_total_pixel
        self.df_global_stats.loc['all classes', 'avg nb class in patch'] = np.mean(nb_class_in_patch)
        self.df_global_stats.loc['all classes', 'avg entropy'] = np.nanmean(list_entropy)

        if self.nbr_classes > 2:
            self.df_global_stats.loc['without last class', 'share multilabel'] /= self.nbr_total_pixel
            self.df_global_stats.loc['without last class', 'avg nb class in patch'] = np.mean(nb_class_in_patch_wlc)
            self.df_global_stats.loc['without last class', 'avg entropy'] = np.nanmean(list_entropy_wlc)

    def compute_stats(self):
        """
        Compute statistics on bands and classes from the data collected during the stage scan_dataset.
        """
        # Statistics on image bands
        for i in range(self.nbr_bands):
            self.df_bands_stats.loc[f'band {i+1}', 'min'] = self.min[i]
            self.df_bands_stats.loc[f'band {i+1}', 'max'] = self.max[i]
            self.df_bands_stats.loc[f'band {i+1}', 'mean'] = self.means[i]
            self.df_bands_stats.loc[f'band {i+1}', 'std'] = self.std[i]
            if self.get_skewness_kurtosis:
                self.df_bands_stats.loc[f'band {i+1}', 'skewness'] = self.skewness[i] / self.nbr_total_pixel
                self.df_bands_stats.loc[f'band {i+1}', 'kurtosis'] = self.kurtosis[i] / self.nbr_total_pixel

        self.zeros_pixels /= self.nbr_total_pixel

        # Divide the histogram binscounts by the number of images in the dataset. Division element wise.
        self.bands_hists = [(band_hist/len(self.dataset)).astype(int) for band_hist in self.bands_hists]

        # Statistics on classes in masks
        for col in self.df_dataset.columns:
            # Ratio of the number of pixels belonging to a class to the total number of pixels in the dataset.
            class_freq = self.df_dataset[col].sum() / self.nbr_total_pixel
            self.df_classes_stats.loc[col, 'pixel freq'] = class_freq
            # For the rest of the stats, if the class is not present in any of the masks then the stats are set to zero

            # Frequency ratio for each class with L1 normalization
            self.df_classes_stats.loc[col, 'regu L1'] = 1 / (class_freq) if class_freq != 0 else 0

            # Frequency ratio for each class with L2 normalization
            self.df_classes_stats.loc[col, 'regu L2'] = \
                1 / np.sqrt((class_freq)) if class_freq != 0 else 0

            # Frequency at which a class is part of at least 5% of an image
            self.df_classes_stats.loc[col, 'freq 5% pixel'] = \
                (self.df_dataset[col][self.df_dataset[col] > 0.05 * self.nbr_pixels_per_patch].count())\
                / len(self.dataset)

            # Area under the Lorenz curve of the pixel distribution by class
            x = self.df_dataset[col]
            self.df_classes_stats.loc[col, 'auc'] = \
                2 * np.sum(np.cumsum(np.sort(x))/np.sum(x))/len(self.dataset) if np.sum(x != 0) else 0

    def to_pixel_input_range(self, value):
        """Pixels of image in the input dataset are normalize to the range 0 to 1.
        This function allows the user to obtain statistics that have values in the range of the input dataset.

        Parameters
        ----------
        value : int
            Input value between [0,1].

        Returns
        -------
        int
            Output value changed according to the bit depth of the input dataset.
        """
        return value * self.depth_dict[self.bit_depth]

    def plot_hist(self, generate=True):
        """Plot histograms with the bands distributions.
        The histograms can be directly plot or save in an image with '.png' format.

        Parameters
        ----------
        generate : bool, optional
            Option to know if the histograms have to be plot or save in an image, by default False

        Returns
        -------
        str
            Path where the output image will be stored.
        """
        if len(self.bins) <= 20:
            labels = []
            for i in range(len(self.bins)):
                if i < len(self.bins) - 1:
                    labels.append(f'{str(self.bins[i])}-{str(self.bins[i+1])}')

        n_plots = self.nbr_bands
        n_cols = 3
        n_rows = ((n_plots - 1) // n_cols) + 1

        plt.figure(figsize=(7 * n_cols, 6 * n_rows))
        for i in range(n_plots):
            plt.subplot(n_rows, n_cols, i+1)
            c = [float(i) / float(self.nbr_bands), 0.0, float(self.nbr_bands-i) / float(self.nbr_bands)]
            plt.bar(range(len(self.bands_hists[i])), self.bands_hists[i], width=0.8, linewidth=2, capsize=20, color=c)
            if len(self.bins) <= 20:
                plt.xticks(range(len(self.bands_hists[i])), labels, rotation=35)
            plt.title(f'Distribution of pixel for band {i+1}')
            plt.xlabel("Pixel bins")
            if i % n_cols == 0:
                plt.ylabel("Pixels count")

        plt.tight_layout(pad=3.0)

        if generate:
            output_path = os.path.join(os.path.dirname(self.output_path), 'stats_hists.png')
            plt.savefig(output_path)
            return output_path
        else:
            plt.show()
