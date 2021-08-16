from odeon.commons.reports.report import Report


class Report_Multiclass(Report):

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
        print(self.input_object.df_thresholds)

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
        metrics_html = f"""
            <h1><center> ODEON  Metrics</center></h1>

            <h2>Macro Strategy</h2>
            <h3>* Metrics</h3>
            {self.df_to_html(self.round_df_values(self.input_object.df_report_macro))}
            <h3>* Confusion Matrix</h3>
            <p><img alt="Macro Confusion Matrix" src={
                self.input_object.plot_confusion_matrix(
                    self.input_object.cm_macro,
                    nbr_class=2,
                    name_plot='cm_macro.png',
                    cmap='summer')} /></p>

            <h2>Micro Strategy</h2>
            <h3>* Metrics</h3>
            {self.df_to_html(self.round_df_values(self.input_object.df_report_micro))}
            <h3>* Confusion Matrix</h3>
            <p><img alt="Macro Confusion Matrix" src={
                self.input_object.plot_confusion_matrix(
                    self.input_object.cm_micro,
                    name_plot='cm_micro.png')} /></p>

            <h2>Per class Strategy</h2>
            <h3>* Metrics</h3>
            {self.df_to_html(self.round_df_values(self.input_object.df_report_classes))}
            <h3>* Calibration Curve</h3>
            <p><img alt="Calibration Curve" src={
                self.input_object.plot_calibration_curve()} /></p>
            <h3>* ROC and PR Curves</h3>
            <p><img alt="ROC and PR Curves" src={
                self.input_object.plot_ROC_PR_per_class()} /></p>
            """
        with open(self.input_object.output_path, "w") as output_file:
            output_file.write(header_html)
            output_file.write(begin_html)
            output_file.write(metrics_html)
            output_file.write(end_html)

            # <h2>Roc Curve</h2>
            # <p><img alt="Roc Curve" src={
            #     self.input_object.plot_ROC_curve(self.input_object.df_thresholds['FPR'],
            #          self.input_object.df_thresholds['Recall'])} /></p>'

            # <h2>Precision Recall Curve</h2>
            # <p><img alt="PR Curve" src={
            #     self.input_object.plot_PR_curve(self.input_object.df_thresholds['Recall'],
            #          self.input_object.df_thresholds['Precision'])} /></p>'

            # <h2>Calibration Curve</h2>
            # <p><img alt="Calibration Curve" src={
            #     self.input_object.plot_calibration_curve()} /></p>'
