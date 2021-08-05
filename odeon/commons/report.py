"""Module allowing the construction of reports for the Statistics and Metrics tools.
The report can be created in a certain format (.json, .md. html) and can also be displayed
directly in the terminal. The created report will be located in the path defined in the
configuration file of the tool on which the report is made.
"""

import os
import json
import numpy as np
import pandas as pd
from odeon import LOGGER
from odeon.commons.logger.logger import get_new_logger, get_simple_handler

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_report")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)


HTML_FILE = 'odeon/commons/jupyter_layout.html'
PADDING = 1
ROUND_DECIMALS = 3


def round_df_values(df, round_decimals=ROUND_DECIMALS):
    """Round the values in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame witth values to export (preferably float)

    Returns
    -------
    df : pd.DataFrame
        DataFrame with rounded values.
    """
    return df.apply(lambda x: pd.to_numeric(x, downcast="float").round(decimals=round_decimals))


def longest(input_list):
    """Return the longest element in a list.

    Parameters
    ----------
    input_list : list
        List of elements, could be a list of string, int or float.

    Returns
    -------
    str
        Longest element element in a list.
    """
    maxlen = max(len(str(s)) for s in input_list)
    longest = filter(lambda s: len(str(s)) == maxlen, input_list)
    return str(list(longest)[0])


def df_to_md(df, input_padding=PADDING):
    """Function export a dataframe in a markdown format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with values to export (preferably float)
    input_padding : int, optional
        Space to add before and after a value in a column, by default PADDING
    round_decimals : int, optional
        Number of decimals , by default ROUND_DECIMALS

    Returns
    -------
    str
        String with the dataframe's content in markdown format.
    """
    global padding
    padding = input_padding

    def add_delta(col_len, input_str):
        """Apply a padding to a string according to the size of a column.

        Parameters
        ----------
        col_len : int
            Length of a column.
        input_str : str/float
            String/value to add left and right padding.

        Returns
        -------
        str
            String with padding.
        """
        if isinstance(input_str, float) and np.isnan(input_str):
            input_str = 'Nan'
        input_str = str(input_str)
        delta = col_len - len(input_str)
        if delta % 2 == 0:
            left, right = delta//2, delta//2
        else:
            left, right = (delta+1)//2, delta//2
        return left * ' ' + input_str + right * ' '

    def get_len_cols(df):
        """Get the length of every column in a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with values to export (preferably float)

        Returns
        -------
        list of int
            List with the columns lengths.
        """
        global padding
        len_cols = [0] * len(df.columns)
        for i, col in enumerate(df.columns):
            len_cols[i] = len(longest(df[col])) + 2 * padding
        return len_cols

    len_cols = get_len_cols(df)

    first_line = '|' + (len(longest(df.index)) + 2 * padding) * ' ' + '|' + \
        '|'.join([add_delta(len_cols[i], x) for i, x in enumerate(df.columns)]) + '|'

    sep = '|' + (len(longest(df.index)) + 2 * padding) * '-' + '|' + \
        '|'.join([len_col * '-' for len_col in len_cols]) + '|'

    output_mk = [first_line, sep]

    for index in df.index:
        line = '|' + add_delta(len(longest(df.index)) + 2 * padding, index) + '|' + \
            '|'.join([add_delta(len_cols[i], df.loc[index, col]) for i, col in enumerate(df.columns)]) + '|'
        output_mk.append(line)

    return '\n'.join(output_mk)


def df_to_html(df):
    """Create from a dataframe a string containing the value of the dataframe in a html table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with values to export (preferably float).

    Returns
    -------
    str
        String with table in html.
    """
    html = "<table>"
    thead = "<thead><tr>" + "<th>" + len(longest(df.index)) * ' ' + "</th>" + \
        ''.join([f"<th>{col}</th>" for col in df.columns]) + "</tr></thead>"
    tbody = "<tbody>"
    for idx in df.index:
        tbody += f"<tr><th>{idx}</th>"
        for col in df.columns:
            tbody += f"<th>{str(df.loc[idx, col])}</th>"
        tbody += "</tr>"
    tbody += "</tbody>"
    return html + thead + tbody + "</table>"


def Report_Factory(input_object):
    """Function assigning which class should be used.

    Parameters
    ----------
    input_object : Statistics/Metrics
        Input object to use to create a report.

    Returns
    -------
    Stats_Report/Metric_Report
        An object making the report.
    """
    reports = {"Statistics": Stats_Report,
               "Metrics": Metric_Report}
    return reports[input_object.__class__.__name__](input_object)


class Report(object):

    def __init__(self, input_object):
        """Init function for the report class

        Parameters
        ----------
        input_object : Statistics/Metrics
            Object on which the report will be made.
            Can be an object Statistics or Metrics from odeon.common.
        """
        self.input_object = input_object
        self.html_file = HTML_FILE

    def create_report(self):
        """Determine the output of the tool. Can create a report in the type request in the
        configuration file. Also can directly display the report in the terminal.
        """
        if self.input_object.output_path is not None:
            ext = os.path.splitext(self.input_object.output_path)[1]
            if ext == '.html':
                self.to_html()
            elif ext == '.json':
                self.to_json()
            elif ext == '.md':
                self.to_md()
            else:
                LOGGER.warning('WARNING: the extension passed as input for the output file is incorrect.\
                    Statistics will be displayed directly in the terminal.')
                self.to_terminal()
        else:
            self.to_terminal()

    def to_terminal(self):
        """Display the results of the tool in the terminal. (WARNING: Matplotlib librairy is use)
        """
        pass

    def to_json(self):
        """Create a report in the json format.
        """
        pass

    def to_md(self):
        """Create a report with the markdown format.
        """
        pass

    def to_html(self):
        """Create a report in the html format.
        """
        pass


class Metric_Report(Report):

    def __init__(self, input_object):
        """Init function of a Metric_Report object.

        Parameters
        ----------
        input_object : Metrics
            Object Metric with the results to export in a report.
        """
        super().__init__(input_object)

    def to_terminal(self):
        """Display the results of the Metrics tool in the terminal.
        """
        pass

    def to_json(self):
        """Create a report in the json format.
        """
        pass

    def to_md(self):
        """Create a report in the markdown format.
        """
        pass

    def to_html(self):
        """Create a report in the html format.
        """
        pass


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
        self.input_object.df_bands_stats = round_df_values(self.input_object.df_bands_stats)
        self.input_object.df_classes_stats = round_df_values(self.input_object.df_classes_stats)
        self.input_object.df_global_stats = round_df_values(self.input_object.df_global_stats)

    def to_terminal(self):
        """
        Display directly in the terminal the dataframes containing the stats and also plot the bands' histograms.
        """
        self.rounded_stats()  # Rounded the values in the dataframes.
        STD_OUT_LOGGER.info("* Images bands statistics: ")
        STD_OUT_LOGGER.info(self.input_object.df_bands_stats)
        STD_OUT_LOGGER.info("* Classes statistics: ")
        STD_OUT_LOGGER.info(self.input_object.df_classes_stats)
        STD_OUT_LOGGER.info("* Global statistics: ")
        STD_OUT_LOGGER.info(self.input_object.df_global_stats)
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
            df_to_md(self.input_object.df_bands_stats) + \
            '\n\n' + \
            '* Statistics computed on the bands of the images: min, max, mean, std for each band.' + \
            '\n\n' + \
            '## Classes statistics' + \
            '\n\n' + \
            df_to_md(self.input_object.df_classes_stats) + \
            '\n\n' + \
            """* Statistics computed on the classes present in the masks:
            - regu L1: Class-Balanced Loss Based on Effective Number of Samples 1/frequency(i)
            - regu L2: Class-Balanced Loss Based on Effective Number of
            Samples 1/sqrt(frequency(i))
            - pixel_freq: Overall share of pixels labeled with a given class.
            - sample_freq_5%pixel: Share of samples with at least
                5% pixels of a given class.
                The lesser, the more concentrated on a few samples a class is.
            - auc: Area under the Lorenz curve of the pixel distribution of a
                given class across samples. The lesser, the more concentrated
                on a few samples a class is. Equals pixel_freq if the class is
                the samples are either full of or empty from the class. Equals
                1 if the class is homogeneously distributed across samples.""" + \
            '\n\n' + \
            '## Global statistics  ' + \
            '\n\n' + \
            df_to_md(self.input_object.df_global_stats) + \
            '\n\n' + \
            """* Statistics computed on the globality of the dataset: (either with all classes or without the
            last class if we are not in the binary case)
            - Percentage of pixels shared by several classes (share_multilabel)
            - the number of classes in an image (avg_nb_class_in_patch)
            - the average entropy (avg_entropy)""" + \
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
            df_to_html(self.input_object.df_bands_stats) + \
            """<p>Statistics computed on the bands of the images: min, max, mean, std for each band</p>

            <h2>Classes statistics</h2>
            """ + \
            df_to_html(self.input_object.df_classes_stats) + \
            """
            <p>Statistics computed on the classes present in the masks:</p>
            <ul>
                <li>regu L1: Class-Balanced Loss Based on Effective Number of Samples 1/frequency(i)</li>
                <li>regu L2: Class-Balanced Loss Based on Effective Number of Samples 1/sqrt(frequency(i))</li>
                <li>pixel_freq: Overall share of pixels labeled with a given class.</li>
                <li>sample_freq_5%pixel: Share of samples with at least 5% pixels of a given class.
                The lesser, the more concentrated on a few samples a class is.</li>
                <li>auc: Area under the Lorenz curve of the pixel distribution of a given class across samples.
                The lesser, the more concentrated on a few samples a class is.
                Equals pixel_freq if the class is the samples are either full of or empty from the class.
                Equals 1 if the class is homogeneously distributed across samples.</li>
            </ul>

            <h2>Global statistics</h2>

            """ + \
            df_to_html(self.input_object.df_global_stats) + \
            """

            <p>Statistics computed on the globality of the dataset: (either with all classes or without the
            last class if we are not in the binary case)</p>

            <ul>
                <li>Percentage of pixels shared by several classes (share_multilabel)</li>
                <li>the number of classes in an image (avg_nb_class_in_patch)</li>
                <li>the average entropy (avg_entropy)</li>
            </ul>

            <h2>Image bands histograms</h2>

            """ + \
            f'<p><img alt="Images bands histograms" src={self.input_object.plot_hist(generate=True)} /></p>'

        with open(self.input_object.output_path, "w") as output_file:
            output_file.write(begin_html)
            output_file.write(stats_html)
            output_file.write(end_html)
