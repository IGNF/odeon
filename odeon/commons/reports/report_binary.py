import os.path as osp
import json
from odeon.commons.reports.report import Report
from odeon.commons.metric.plots import plot_norm_and_value_cms, plot_confusion_matrix,\
     plot_calibration_curves, plot_roc_pr_curves, plot_hists


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
        if self.input_object.output_type != 'json':
            conf_mat = self.input_object.cms[self.input_object.threshold]
            if self.input_object.get_normalize:
                self.cm = plot_norm_and_value_cms(conf_mat,
                                                  labels=self.input_object.class_labels,
                                                  output_path=osp.join(self.output_path, 'cm_binary.png'),
                                                  per_class_norm=False)
            else:
                self.cm = plot_confusion_matrix(conf_mat,
                                                labels=self.input_object.class_labels,
                                                output_path=osp.join(self.output_path, 'cm_binary.png'))

            if self.input_object.get_calibration_curves:
                self.calibration_curve = plot_calibration_curves(prob_true=self.input_object.prob_true,
                                                                 prob_pred=self.input_object.prob_pred,
                                                                 hist_counts=self.input_object.hist_counts,
                                                                 labels=self.input_object.class_labels[0],
                                                                 nbr_class=1,
                                                                 output_path=osp.join(self.output_path,
                                                                                      'calibration_curves.png'),
                                                                 bins=self.input_object.bins,
                                                                 bins_xticks=self.input_object.bins_xticks)

            if self.input_object.get_ROC_PR_curves:
                self.ROC_PR_curve = plot_roc_pr_curves(data=self.input_object.vect_curves,
                                                       labels=self.input_object.class_labels[0],
                                                       nbr_class=1,
                                                       output_path=osp.join(self.output_path, 'roc_pr_curves.png'),
                                                       decimals=self.input_object.decimals)

            if self.input_object.get_hists_per_metrics:
                self.metrics_hists = plot_hists(hists=self.input_object.hists_metrics,
                                                list_metrics=self.input_object.metrics_names[:-1],
                                                output_path=osp.join(self.output_path, 'hists_metrics.png'),
                                                n_bins=self.input_object.n_bins,
                                                bins_xticks=self.input_object.bins_xticks,
                                                n_cols=3, size_col=7, size_row=6)

    def to_json(self):
        """Create a report in the json format.
        """
        dict_export = self.input_object.dict_export
        dict_export['metrics report'] = self.round_df_values(self.input_object.df_report_metrics,
                                                             to_percent=True).T.to_dict()
        cms_json = {}
        for threshold in self.input_object.cms:
            cms_json[threshold] = self.input_object.cms[threshold].tolist()

        dict_export['cms'] = cms_json

        json_object = json.dumps(dict_export)
        with open(osp.join(self.output_path, 'report_metrics.json'), "w") as output_file:
            output_file.write(json_object)

    def to_md(self):
        """Create a report in the markdown format.
        """
        md_main = f"""
# ODEON - Metrics

## Main metrics

{self.df_to_md(self.round_df_values(self.input_object.df_report_metrics,
                                    to_percent=True))}

* Metrics computed with a threshod of : {self.input_object.threshold}

## Confusion matrix
![Confusion matrix](./{osp.basename(self.cm)})
"""

        md_elements = [md_main]

        if self.input_object.get_ROC_PR_curves:
            roc_pr_curves = f"""
## Roc Curve
![Roc curve](./{osp.basename(self.ROC_PR_curve)})
"""
            md_elements.append(roc_pr_curves)

        if self.input_object.get_calibration_curves:
            calibration_curves = f"""
## Calibration Curve
![Calibration curve](./{osp.basename(self.calibration_curve)})
"""
            md_elements.append(calibration_curves)

        if self.input_object.get_hists_per_metrics:
            metrics_histograms = f""""
## Metrics Histograms
![Histograms per metric](./{osp.basename(self.metrics_hists)})
"""
            md_elements.append(metrics_histograms)

        with open(osp.join(self.output_path, 'binary_metrics.md'), "w") as output_file:
            for md_element in md_elements:
                output_file.write(md_element)

    def to_html(self):
        """Create a report in the html format.
        """
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
            {self.df_to_html(self.round_df_values(self.input_object.df_report_metrics, to_percent=True))}
            <p>Metrics computed with a threshod of : {self.input_object.threshold}</p>

            <h2>Confusion Matrix</h2>
            <p><img alt="Confusion Matrix" src=./{osp.basename(self.cm)} /></p>"""

        html_elements = [header_html, self.begin_html, main_html]

        if self.input_object.get_ROC_PR_curves:
            roc_pr_curves = f"""
            <h2>Roc Curve</h2>
            <p><img alt="Roc Curve" src=./{osp.basename(self.ROC_PR_curve)} /></p>
            """
            html_elements.append(roc_pr_curves)

        if self.input_object.get_calibration_curves:
            calibration_curves = f"""
                <h2>Calibration Curve</h2>
                <p><img alt="Calibration Curve" src=./{osp.basename(self.calibration_curve)} /></p>
                """
            html_elements.append(calibration_curves)

        if self.input_object.get_hists_per_metrics:
            metrics_histograms = f"""
                <h2>Metrics Histograms</h2>
                <p><img alt="Metrics Histograms" src=./{osp.basename(self.metrics_hists)} /></p>
                """
            html_elements.append(metrics_histograms)

        html_elements.append(end_html)

        with open(osp.join(self.output_path, 'metrics_binary.html'), "w") as output_file:
            for html_part in html_elements:
                output_file.write(html_part)
