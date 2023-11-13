"""Module allowing the construction of reports for the Statistics and Metrics tools.
The report can be created in a certain format (.json, .md. html) and can also be displayed
directly in the terminal. The created report will be located in the path defined in the
configuration file of the tool on which the report is made.
"""
import os
import numpy as np
import pandas as pd
from odeon.commons.logger.logger import get_new_logger, get_simple_handler

HTML_FILE = 'jupyter_layout.html'
PADDING = 1


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
        self.padding = PADDING
        self.round_decimals = self.input_object.decimals
        self.output_path = input_object.output_path
        if self.input_object.output_type == 'html':
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, self.html_file), "r") as reader:
                self.begin_html = reader.read()

        " A logger for big message "
        self.STD_OUT_LOGGER = get_new_logger("stdout_report")
        ch = get_simple_handler()
        self.STD_OUT_LOGGER.addHandler(ch)

    def create_data(self):
        pass

    def create_report(self):
        """Determine the output of the tool. Can create a report in the type request in the
        configuration file. Also can directly display the report in the terminal.
        """
        self.create_data()
        if self.input_object.output_type == 'html':
            self.to_html()
        elif self.input_object.output_type == 'json':
            self.to_json()
        elif self.input_object.output_type == 'md':
            self.to_md()
        else:
            self.to_terminal()

    def round_df_values(self, df, round_decimals=None, to_percent=False):
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
        if round_decimals is None:
            round_decimals = self.round_decimals
        if to_percent:
            return df.apply(lambda x: pd.to_numeric(x * 100, downcast="float").round(decimals=round_decimals - 2))
        else:
            return df.apply(lambda x: pd.to_numeric(x, downcast="float").round(decimals=round_decimals))

    def longest(self, input_list):
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

    def df_to_md(self, df):
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
        padding = self.padding

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
                len_cols[i] = len(self.longest(df[col])) + 2 * padding
            return len_cols

        len_cols = get_len_cols(df)

        first_line = '|' + (len(self.longest(df.index)) + 2 * padding) * ' ' + '|' + \
            '|'.join([add_delta(len_cols[i], x) for i, x in enumerate(df.columns)]) + '|'

        sep = '|' + (len(self.longest(df.index)) + 2 * padding) * '-' + '|' + \
            '|'.join([len_col * '-' for len_col in len_cols]) + '|'

        output_mk = [first_line, sep]

        for index in df.index:
            line = '|' + add_delta(len(self.longest(df.index)) + 2 * padding, index) + '|' + \
                '|'.join([add_delta(len_cols[i], df.loc[index, col]) for i, col in enumerate(df.columns)]) + '|'
            output_mk.append(line)

        return '\n'.join(output_mk)

    def df_to_html(self, df):
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
        thead = "<thead><tr>" + "<th>" + len(self.longest(df.index)) * ' ' + "</th>" + \
            ''.join([f"<th>{col}</th>" for col in df.columns]) + "</tr></thead>"
        tbody = "<tbody>"
        for idx in df.index:
            tbody += f"<tr><th>{idx}</th>"
            for col in df.columns:
                tbody += f"<th>{str(df.loc[idx, col])}</th>"
            tbody += "</tr>"
        tbody += "</tbody>"
        return html + thead + tbody + "</table>"

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
