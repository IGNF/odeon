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
from cycler import cycler
from odeon.commons.reports.report_factory import Report_Factory
from odeon.commons.exception import OdeonError, ErrorCodes

BATCH_SIZE = 1
NUM_WORKERS = 1
BIT_DEPTH = '8 bits'
GET_SKEWNESS_KURTOSIS = False
GET_RADIO_STATS = False
MOVING_AVERAGE = 3
DECIMALS = 3


class Statistics():

    def __init__(self,
                 dataset,
                 output_path,
                 output_type=None,
                 bands_labels=None,
                 class_labels=None,
                 get_skewness_kurtosis=GET_SKEWNESS_KURTOSIS,
                 bit_depth=BIT_DEPTH,
                 bins=None,
                 nbr_bins=None,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS,
                 get_radio_stats=GET_RADIO_STATS,
                 plot_stacked=False):
        """
        Init function of Statistics class.

        Parameters
        ----------
        dataset :  PatchDataset
            Dataset from odeon.nn.datasets which contains the images and masks.
        output_path: str
            Path where the report with the computed statistics will be created.
        output_type : str, optional
            Desired format for the output file. Could be json, md or html.
            A report will be created if the output type is html or md.
            If the output type is json, all the data will be exported in a dict in order
            to be easily reusable, by default html.
        bands_labels : list of str, optional
            Label for each bands in the dataset, by default None.
        class_labels : list of str, optional
            Label for each class in the dataset, by default None.
        bins: list, optional
            List of the bins to build the histograms of the image bands, by default None.
        nbr_bins: int, optional
            If bins is not given in input, the list of bins will be created with the
            parameter nbr_bins defined here. If None the bins will be automatically
            defined according to the maximum value of the pixels in the dataset, by default None.
        get_skewness_kurtosis: bool
            Boolean to compute or not skewness and kurtosis, by default False.
        bit_depth: str, optional
            The number of bits used to represent each pixel in an image, , by default "8 bits".
        batch_size: int
            The number of image in a batch, by default 1.
        num_workers: int, optional
            Number of workers to use in the pytorch dataloader, by default 1.
        get_radio_stats: bool, optional
            Bool to compute radiometry statistics, i.e. the distribution of each image's band according
            to each class, by default True.
        plot_stacked: bool, optional
            Parameter to know if the histograms of each band should be displayed on the same figure
            or on different figures, by default False.
        """
        # Input arguments
        self.dataset = dataset

        if not os.path.exists(output_path):
            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"Output folder ${output_path} does not exist.")
        else:
            self.output_path = output_path

        if output_type in ['md', 'json', 'html']:
            self.output_type = output_type
        else:
            LOGGER.error('ERROR: the output file can only be in md, json, html')
            self.output_type = 'html'
        self.decimals = DECIMALS
        self.nbr_bands = len(self.dataset.image_bands)
        self.nbr_classes = len(self.dataset.mask_bands)
        self.nbr_pixels_per_patch = self.dataset.height * self.dataset.width
        self.nbr_total_pixel = len(self.dataset) * self.nbr_pixels_per_patch

        if bands_labels is not None and len(bands_labels) != self.nbr_bands:
            LOGGER.error('ERROR: parameter bands_labels should have a number of values equal to the number of bands .')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input parameter bands_labels is incorrect.")
        elif bands_labels is None:
            self.bands_labels = [f'band {i}' for i in range(1, self.nbr_bands+1)]
        else:
            self.bands_labels = bands_labels

        if class_labels is not None and len(class_labels) != self.nbr_classes:
            LOGGER.error('ERROR: parameter class_labels should have a number of values equal to the number of classes.')
            raise OdeonError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input parameter class_labels is incorrect.")
        elif class_labels is None:
            self.class_labels = [f'class {i}' for i in range(1, self.nbr_classes+1)]
        else:
            self.class_labels = class_labels

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

        # Labels for histograms
        if len(self.bins) <= 20:
            self.labels = []
            for i in range(len(self.bins)):
                if i < len(self.bins) - 1:
                    self.labels.append(f'{str(self.bins[i])}-{str(self.bins[i+1])}')

        # Stats radiometry
        self.get_radio_stats = get_radio_stats
        if self.get_radio_stats:
            self.df_radio = pd.DataFrame(index=self.class_labels, columns=self.bands_labels, dtype=object)
            self.df_radio = self.df_radio.applymap(lambda x: np.zeros(len(self.bins) - 1))

        self.plot_stacked = plot_stacked

        # Dataframes creation
        self.df_dataset, self.df_bands_stats, self.df_classes_stats, self.df_global_stats, self.bands_hists =\
            self.create_data_for_stats()

    def run(self):
        """
        Run the methods to compute metrics.
        """
        self.scan_dataset()
        self.compute_stats()
        self.report = Report_Factory(self)

    def __call__(self):
        """
        Function to generate an output file when the instance is called.
        """
        self.run()
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
        if bins is None and self.nbr_bins is not None:
            bins = [round((i/self.nbr_bins) * max_pixel_value, 3) for i in range(self.nbr_bins)]
            bins.append(max_pixel_value)

        elif bins is None and self.nbr_bins is None:
            bins = np.arange(start=0, stop=self.depth_dict[self.bit_depth] + 1, step=MOVING_AVERAGE)

        if isinstance(bins, list):
            bins = np.array(bins)

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

        df_dataset = pd.DataFrame(index=range(len(self.dataset)), columns=self.class_labels)

        if self.get_skewness_kurtosis:
            header_bands = ['min', 'max', 'mean', 'std', 'skewness', 'kurtosis']
        else:
            header_bands = ['min', 'max', 'mean', 'std']

        df_bands_stats = pd.DataFrame(index=self.bands_labels, columns=header_bands)

        df_classes_stats = pd.DataFrame(index=self.class_labels,
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

                for idx_class, name_class in enumerate(self.class_labels):
                    vect_class = mask[:, :, idx_class].flatten()
                    self.df_dataset.loc[index, name_class] = np.count_nonzero(vect_class)
                index += 1

                # Information storage for statistics.
                self.df_global_stats.loc['all classes', 'share multilabel'] += \
                    np.count_nonzero(np.sum(mask.copy(), axis=2) > 1)
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
                         np.count_nonzero(np.sum(mask[:, :, :(self.nbr_classes - 1)], axis=2) > 1)
                    nb_class_in_patch_wlc.append(np.count_nonzero(vect_sum_class[:-1]))

                    if all(np.equal(vect_sum_class[:-1], np.zeros(self.nbr_classes-1))):
                        sample_entropy_wlc = 0
                    else:
                        vect_normalize_wlc = vect_sum_class[:-1]/self.nbr_pixels_per_patch
                        sample_entropy_wlc = entropy(vect_normalize_wlc)
                    list_entropy_wlc.append(sample_entropy_wlc)

                # Make histogram of image band values where a class is present in the mask.
                if self.get_radio_stats:
                    image, mask = image.astype(np.int64), mask.astype(np.int64)
                    for i, class_i in enumerate(self.class_labels):
                        for j, band_j in enumerate(self.bands_labels):
                            img_filter = image[:, :, j][mask[:, :, i] == 1]
                            self.df_radio.loc[class_i, band_j] += np.histogram(img_filter,
                                                                               bins=self.bins)[0]

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
        for i, name_band in enumerate(self.bands_labels):
            self.df_bands_stats.loc[name_band, 'min'] = self.min[i]
            self.df_bands_stats.loc[name_band, 'max'] = self.max[i]
            self.df_bands_stats.loc[name_band, 'mean'] = self.means[i]
            self.df_bands_stats.loc[name_band, 'std'] = self.std[i]
            if self.get_skewness_kurtosis:
                self.df_bands_stats.loc[name_band, 'skewness'] = self.skewness[i] / self.nbr_total_pixel
                self.df_bands_stats.loc[name_band, 'kurtosis'] = self.kurtosis[i] / self.nbr_total_pixel

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

    def plot_hists(self, bincounts, n_cols=3, size_row=6, size_col=6, name_plot=None, display_stats=False):
        """
        Plot histograms from bincounts.
        The histograms are saved in an image with '.png' format.
        For the figure plot, the distributions can be displayed independently each in a figure
        or the distributions can be grouped under the same figure to better visualize their differences.

        Parameters
        ----------
        bincounts : np.array
            List of arrays containing for each the number of obersation counted in each bins.
        n_cols : int, optional
            [description], by default 3
        size_row : int, optional
            Size of a row in the figure, by default 6
        size_col : int, optional
            Sive of a column in the figure, by default 6
        name_plot : str, optional
            Name to give to the ouput .png image, by default None
        display_stats : bool, optional
            Bool to know if stats (mean/std) have to be inserted in the plots, by default False

        Returns
        -------
        str
            Path where the output image will be stored.
        """
        default_cycler = cycler(color=['tab:red', 'tab:green', 'tab:blue', 'darkviolet', 'darkolivegreen',
                                       'orange', 'coral', 'crimson', 'darkmagenta', 'midnightblue', 'cadetblue'])
        plt.rc('axes', prop_cycle=default_cycler)

        if not self.plot_stacked:
            n_plot = self.nbr_bands
            n_rows = ((n_plot - 1) // n_cols) + 1
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(size_col * n_cols, size_row * n_rows))
            axes = axes.ravel()
            axes_to_del = len(axes) - n_plot
            for i, ax_prop in enumerate(zip(self.bands_labels, default_cycler)):
                band_label, c = ax_prop[0], ax_prop[1]
                bincount = bincounts[i]
                if display_stats:
                    mean = np.round(self.df_bands_stats.loc[band_label, "mean"], self.decimals)
                    std = np.round(self.df_bands_stats.loc[band_label, "std"], self.decimals)
                    axes[i].axvline(mean, label=f"Mean: {str(mean)}", linestyle='--', alpha=0.5)
                    axes[i].axvspan(mean - std, mean + std, label=f"Std: {str(std)}",
                                    linestyle='--', alpha=0.5, color="lightblue")
                    axes[i].legend(loc='upper right')
                axes[i].hist(self.bins[:-1],
                             weights=bincount,
                             bins=self.bins,
                             color=c['color'],
                             alpha=0.7)
                axes[i].set_ylabel("Pixel count")
                axes[i].set_xlabel("Pixel distribution")
                if len(self.bins) <= 20:
                    axes[i].set_xticks(range(len(self.labels)))
                    axes[i].set_xticklabels(self.labels)
                    plt.setp(axes[i].get_xticklabels(), rotation=35)
                axes[i].set_title(f"{band_label.capitalize()}", fontsize=13)
                axes[i].grid(b=True, which='major', linestyle='-')
                axes[i].minorticks_on()
                axes[i].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            if axes_to_del != 0:
                for i in range(axes_to_del):
                    fig.delaxes(axes[n_plot + i])
        else:
            plt.figure(figsize=(12, 6))
            for i, plot_prop in enumerate(zip(self.bands_labels, default_cycler)):
                band_label, color = plot_prop[0], plot_prop[1]['color']
                # if display_stats:
                #     mean = np.around(self.df_bands_stats.loc[band_label, "mean"], decimals=self.decimals)
                #     plt.axvline(mean, label=f"Mean {band_label}: {str(mean)}",
                #                 linestyle='--', alpha=0.5, color=color)
                plt.hist(self.bins[:-1], weights=bincounts[i], bins=self.bins,
                         histtype='step', label=band_label, alpha=0.7, color=color)
            plt.grid(b=True, which='major', linestyle='-')
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylabel("Pixel count")
            plt.xlabel("Pixel distribution")

        plt.tight_layout(pad=3)
        output_path = os.path.join(self.output_path, name_plot)
        plt.savefig(output_path)
        return output_path

    def plot_hists_bands(self, name_plot='stats_hists.png'):
        """Plot histograms for the image bands distributions.
        The histograms are saved in an image with '.png' format.

        Returns
        -------
        dict
            Dict with paths where the output images will be stored.
        """
        bincounts = [self.bands_hists[i] for i in range(self.nbr_bands)]
        return self.plot_hists(bincounts, name_plot=name_plot, display_stats=True)

    def plot_hists_radiometry(self):
        """Plot histograms representing the distribution of pixel values in each band when a specific class is present.

        Returns
        -------
        dict
            Dict with paths where the output images will be stored.
        """
        output_paths = {}
        for class_i in self.class_labels:
            bincounts = [self.df_radio.loc[class_i, band_label] for band_label in self.bands_labels]
            output_paths[class_i] = self.plot_hists(bincounts, name_plot=f"radio_{'_'.join(class_i.split(' '))}.png")
        return output_paths
