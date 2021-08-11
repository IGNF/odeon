import json
from odeon.commons.report import Report


class Stats_Report(Report):
    """Class to make report with the results of an object Statistics.
    """
    def __init__(self, input_object):
        """Init function of the Stats_Report object.

        Parameters
        ----------
        input_object : [type]
            [description]
        """
        super().__init__(input_object)

    def rounded_stats(self):
        """
        Statistics object with rounded values dataframes.
        """
        self.input_object.df_bands_stats = self.round_df_values(self.input_object.df_bands_stats)
        self.input_object.df_classes_stats = self.round_df_values(self.input_object.df_classes_stats)
        self.input_object.df_global_stats = self.round_df_values(self.input_object.df_global_stats)

    def to_terminal(self):
        """
        Display directly in the terminal the dataframes containing the stats and also plot the bands' histograms.
        """
        self.rounded_stats()  # Rounded the values in the dataframes.
        self.STD_OUT_LOGGER.info("* Images bands statistics: ")
        self.STD_OUT_LOGGER.info(self.input_object.df_bands_stats)
        self.STD_OUT_LOGGER.info("* Classes statistics: ")
        self.STD_OUT_LOGGER.info(self.input_object.df_classes_stats)
        self.STD_OUT_LOGGER.info("* Global statistics: ")
        self.STD_OUT_LOGGER.info(self.input_object.df_global_stats)
        self.input_object.plot_hist()

    def to_json(self):
        """
        Generate a JSON output file at the output path pass in the initialization of the instance.
        """
        data_to_dict = {'bands': self.input_object.df_bands_stats.T.to_dict(),
                        'classes': self.input_object.df_classes_stats.T.to_dict(),
                        'globals': self.input_object.df_global_stats.T.to_dict(),
                        'hists': {'bins': self.input_object.bins,
                                  'bins_counts':
                                  {f'band {i+1}': [int(x) for x in hist]
                                   for i, hist in enumerate(self.input_object.bands_hists)}}}

        with open(self.input_object.output_path, 'w') as output_file:
            json.dump(data_to_dict, output_file, indent=4)

    def to_md(self):
        """Create a report in the markdown format.
        """
        self.rounded_stats()  # Rounded the values in the dataframes.

        md_text = '# ODEON - Statistics' + \
            '\n\n' + \
            '## Image bands statistics' + \
            '\n\n' + \
            self.df_to_md(self.input_object.df_bands_stats) + \
            '\n\n' + \
            '* Statistics computed on the bands of the images: min, max, mean, std for each band.' + \
            f'* Percent of zeros pixels in dataset images: \
                {round(self.input_object.zeros_pixels, self.round_decimals)} %.' + \
            '\n\n' + \
            '## Classes statistics' + \
            '\n\n' + \
            self.df_to_md(self.input_object.df_classes_stats) + \
            '\n\n' + \
            """* Statistics computed on the classes present in the masks:
            - regu L1: Class-Balanced Loss Based on Effective Number of Samples 1/frequency(i)
            - regu L2: Class-Balanced Loss Based on Effective Number of
            Samples 1/sqrt(frequency(i))
            - pixel freq: Overall share of pixels labeled with a given class.
            - sample freq 5%pixel: Share of samples with at least
                5% pixels of a given class.
                The lesser, the more concentrated on a few samples a class is.
            - auc: Area under the Lorenz curve of the pixel distribution of a
                given class across samples. The lesser, the more concentrated
                on a few samples a class is. Equals pixel freq if the class is
                the samples are either full of or empty from the class. Equals
                1 if the class is homogeneously distributed across samples.""" + \
            '\n\n' + \
            '## Global statistics  ' + \
            '\n\n' + \
            self.df_to_md(self.input_object.df_global_stats) + \
            '\n\n' + \
            """* Statistics computed on the globality of the dataset: (either with all classes or without the
            last class if we are not in the binary case)
            - Percentage of pixels shared by several classes (share multilabel)
            - the number of classes in an image (avg nb class in patch)
            - the average entropy (avg entropy)""" + \
            '\n\n' + \
            f'![Images bands histograms]({self.input_object.plot_hist(generate=True)})'

        with open(self.input_object.output_path, "w") as output_file:
            output_file.write(md_text)

    def to_html(self):
        """Create a report in the html format.
        """
        self.rounded_stats()
        # TODO: the use of html is not really clean and should be change in the future.
        with open(self.html_file, "r") as reader:
            begin_html = reader.read()

        end_html = '</div></div></body></html>'

        stats_html = """
            <h1><center> ODEON  Statistics</center></h1>

            <h2>Image bands statistics</h2>

            """ + \
            self.df_to_html(self.input_object.df_bands_stats) + \
            f"""<p>Statistics computed on the bands of the images: min, max, mean, std for each band.</p>
            <p>Percent of zeros pixels in dataset images: {
                round(self.input_object.zeros_pixels, self.round_decimals)} %.</p>

            <h2>Classes statistics</h2>
            """ + \
            self.df_to_html(self.input_object.df_classes_stats) + \
            """
            <p>Statistics computed on the classes present in the masks:</p>
            <ul>
                <li>regu L1: Class-Balanced Loss Based on Effective Number of Samples 1/frequency(i)</li>
                <li>regu L2: Class-Balanced Loss Based on Effective Number of Samples 1/sqrt(frequency(i))</li>
                <li>pixel freq: Overall share of pixels labeled with a given class.</li>
                <li>sample freq 5%pixel: Share of samples with at least 5% pixels of a given class.
                The lesser, the more concentrated on a few samples a class is.</li>
                <li>auc: Area under the Lorenz curve of the pixel distribution of a given class across samples.
                The lesser, the more concentrated on a few samples a class is.
                Equals pixel freq if the class is the samples are either full of or empty from the class.
                Equals 1 if the class is homogeneously distributed across samples.</li>
            </ul>

            <h2>Global statistics</h2>

            """ + \
            self.df_to_html(self.input_object.df_global_stats) + \
            """

            <p>Statistics computed on the globality of the dataset: (either with all classes or without the
            last class if we are not in the binary case)</p>

            <ul>
                <li>Percentage of pixels shared by several classes (share multilabel)</li>
                <li>the number of classes in an image (avg nb class in patch)</li>
                <li>the average entropy (avg entropy)</li>
            </ul>

            <h2>Image bands histograms</h2>

            """ + \
            f'<p><img alt="Images bands histograms" src={self.input_object.plot_hist(generate=True)} /></p>'

        with open(self.input_object.output_path, "w") as output_file:
            output_file.write(begin_html)
            output_file.write(stats_html)
            output_file.write(end_html)
