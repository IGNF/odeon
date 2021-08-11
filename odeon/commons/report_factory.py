from odeon.commons.stats_report import Stats_Report
from odeon.commons.bc_metrics_report import BC_Metrics_Report


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
               "BC_Metrics": BC_Metrics_Report}
    return reports[input_object.__class__.__name__](input_object)
