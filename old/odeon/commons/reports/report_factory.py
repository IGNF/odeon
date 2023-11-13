from odeon.commons.reports.report_stats import Report_Stats
from odeon.commons.reports.report_binary import Report_Binary
from odeon.commons.reports.report_multiclass import Report_Multiclass


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
    reports = {"Statistics": Report_Stats,
               "MetricsBinary": Report_Binary,
               "MetricsMulticlass": Report_Multiclass}
    return reports[input_object.__class__.__name__](input_object)
