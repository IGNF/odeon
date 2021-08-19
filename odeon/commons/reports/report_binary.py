import os
from odeon.commons.reports.report import Report


class Report_Binary(Report):

    def __init__(self, input_object):
        """Init function of a Metric_Report object.

        Parameters
        ----------
        input_object : Metrics
            Object Metric with the results to export in a report.
        """
        super().__init__(input_object)

    def create_data(self):
        if self.input_object.output_type == 'terminal':
            self.generate = False
        else:
            self.generate = True

        self.cm = self.input_object.plot_confusion_matrix(self.input_object.cms[self.input_object.threshold],
                                                          labels=['Positive', 'Negative'],
                                                          name_plot='cm_binary.png',
                                                          generate=self.generate)

        if self.input_object.get_calibration_curves and not self.input_object.type_prob == 'hard':
            self.calibration_curve = self.input_object.plot_calibration_curve(generate=self.generate)

        if self.input_object.get_ROC_PR_curves:
            self.ROC_curve = self.input_object.plot_ROC_curve(self.input_object.df_thresholds['FPR'],
                                                              self.input_object.df_thresholds['Recall'],
                                                              generate=self.generate)
            self.PR_curve = self.input_object.plot_PR_curve(self.input_object.df_thresholds['Recall'],
                                                            self.input_object.df_thresholds['Precision'],
                                                            generate=self.generate)

        if self.input_object.get_hists_per_metrics:
            self.metrics_hists = self.input_object.plot_dataset_metrics_histograms(generate=self.generate)

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
        with open(self.html_file, "r") as reader:
            begin_html = reader.read()

        header_html = """
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <title>ODEON Metrics</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js">
        </script>
        """

        end_html = '</div></div></body></html>'

        main_html = f"""
            <h1><center> ODEON  Metrics</center></h1>

            <h2>Main Metrics</h2>
            {self.df_to_html(self.round_df_values(self.input_object.df_report_metrics))}
            <p>Metrics computed with a threshod of : {self.input_object.threshold}</p>

            <h2>Confusion Matrix</h2>
            <p><img alt="Confusion Matrix" src=./{os.path.basename(self.cm)} /></p>"""

        html_elements = [header_html, begin_html, main_html]

        if self.input_object.get_calibration_curves and not self.input_object.type_prob == 'hard':
            calibration_curves = f"""
                <h2>Calibration Curve</h2>
                <p><img alt="Calibration Curve" src=./{os.path.basename(self.calibration_curve)} /></p>
                """
            html_elements.append(calibration_curves)

        if self.input_object.get_ROC_PR_curves:
            roc_pr_curves = f"""
            <h2>Roc Curve</h2>
            <p><img alt="Roc Curve" src=./{os.path.basename(self.ROC_curve)} /></p>

            <h2>Precision Recall Curve</h2>
            <p><img alt="PR Curve" src=./{os.path.basename(self.PR_curve)} /></p>
            """
            html_elements.append(roc_pr_curves)

        if self.input_object.get_hists_per_metrics:
            metrics_histograms = f""""
                <h2>Metrics Histograms</h2>
                <p><img alt="Metrics Histograms" src=./{os.path.basename(self.metrics_hists)} /></p>
                """
            html_elements.append(metrics_histograms)

        html_elements.append(end_html)

        with open(os.path.join(self.input_object.output_path, 'metrics_binary.html'), "w") as output_file:
            for html_part in html_elements:
                output_file.write(html_part)
