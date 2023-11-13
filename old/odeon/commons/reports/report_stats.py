import os
import json
import numpy as np
from odeon.commons.reports.report import Report


class Report_Stats(Report):
    """Class to make report with the results of an object Statistics.
    """
    def __init__(self, input_object):
        """Init function of the Stats_Report object.

        Parameters
        ----------
        input_object : Statistics
            Object Statistics from odeon.commons.statistics.
        """
        super().__init__(input_object)

    def rounded_stats(self):
        """
        Statistics object with rounded values dataframes.
        """
        self.input_object.df_bands_stats = self.round_df_values(self.input_object.df_bands_stats)
        self.input_object.df_classes_stats = self.round_df_values(self.input_object.df_classes_stats)
        self.input_object.df_global_stats = self.round_df_values(self.input_object.df_global_stats)

    def to_json(self):
        """
        Generate a JSON output file at the output path pass in the initialization of the instance.
        """
        data_to_dict = {'bands': self.input_object.df_bands_stats.T.to_dict(),
                        'classes': self.input_object.df_classes_stats.T.to_dict(),
                        'globals': self.input_object.df_global_stats.T.to_dict(),
                        'hists': {'bins': self.input_object.bins.tolist(),
                                  'bins_counts':
                                  {f'band {i+1}': hist.astype(int).tolist()
                                   for i, hist in enumerate(self.input_object.bands_hists)}}}

        if self.input_object.get_radio_stats:
            df = self.input_object.df_radio
            for class_i in self.input_object.class_labels:
                for band_i in self.input_object.bands_labels:
                    if isinstance(df.loc[class_i, band_i], np.ndarray):
                        df.loc[class_i, band_i] = df.loc[class_i, band_i].tolist()
            data_to_dict['radiometry'] = df.T.to_dict()

        with open(os.path.join(self.input_object.output_path, 'stats_report.json'), "w") as output_file:
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
            - the average entropy (avg entropy)

            ## Images bands histograms
            """ + \
            '\n\n' + \
            f'![Images bands histograms](./{os.path.basename(self.input_object.plot_hists_bands())})'

        parts_md = [md_text]

        if self.input_object.get_radio_stats:
            self.path_radios = self.input_object.plot_hists_radiometry()
            radio_class_md = []
            radio_begin = """

## Radiometry per class"""
            radio_class_md.append(radio_begin)
            for class_name in self.input_object.class_labels:
                class_md = f"""
### {class_name.capitalize()} :
![Histograms radio {class_name}](./{os.path.basename(os.path.basename(self.path_radios[class_name]))})"""
                radio_class_md.append(class_md)
            parts_md.append("\n".join(radio_class_md))

        with open(os.path.join(self.input_object.output_path, 'stats_report.md'), "w") as output_file:
            for part_md in parts_md:
                output_file.write(part_md)

    def to_html(self):
        """Create a report in the html format.
        """
        self.rounded_stats()

        header_html = """
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <title>ODEON Statistics</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js">
        </script>
        """
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
            f"""

            <p>Statistics computed on the globality of the dataset: (either with all classes or without the
            last class if we are not in the binary case)</p>

            <ul>
                <li>Percentage of pixels shared by several classes (share multilabel)</li>
                <li>the number of classes in an image (avg nb class in patch)</li>
                <li>the average entropy (avg entropy)</li>
            </ul>

            <h2>Image bands histograms</h2>

            <p><img alt='Images bands histograms' src=./{os.path.basename(self.input_object.plot_hists_bands())} /></p>

            """

        parts_html = [header_html, self.begin_html, stats_html]
        if self.input_object.get_radio_stats:
            self.path_radios = self.input_object.plot_hists_radiometry()
            radio_class_html = []
            radio_begin = """
            <h2>Radiometry per class</h2>
            """
            radio_class_html.append(radio_begin)
            for class_name in self.input_object.class_labels:
                class_html = f"""
            <h3>{class_name.capitalize()} :</h3>
            <p><img alt="Histograms {class_name}" src=./{os.path.basename(self.path_radios[class_name])} /></p>
                """
                radio_class_html.append(class_html)
            parts_html.append("\n".join(radio_class_html))

        parts_html.append(end_html)
        with open(os.path.join(self.input_object.output_path, 'stats_report.html'), "w") as output_file:
            for part_html in parts_html:
                output_file.write(part_html)
