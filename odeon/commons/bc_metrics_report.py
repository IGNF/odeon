from odeon.commons.report import Report


class BC_Metrics_Report(Report):

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
        print(self.input_object.df_roc_metrics)

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

