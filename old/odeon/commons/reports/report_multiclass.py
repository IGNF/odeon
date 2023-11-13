import os.path as osp
import json
from odeon.commons.reports.report import Report
from odeon.commons.metric.plots import plot_norm_and_value_cms, plot_confusion_matrix,\
     plot_calibration_curves, plot_roc_pr_curves


class Report_Multiclass(Report):

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
            if self.input_object.get_normalize:
                self.path_cm_macro = plot_norm_and_value_cms(self.input_object.cm_macro,
                                                             labels=self.input_object.class_labels,
                                                             output_path=osp.join(self.output_path, 'cm_macro.png'))
                self.path_cm_micro = plot_norm_and_value_cms(self.input_object.cm_micro,
                                                             labels=['Positive', 'Negative'],
                                                             output_path=osp.join(self.output_path, 'cm_micro.png'),
                                                             per_class_norm=False)
            else:
                self.path_cm_macro = plot_confusion_matrix(self.input_object.cm_macro,
                                                           labels=self.input_object.class_labels,
                                                           output_path=osp.join(self.output_path, 'cm_macro.png'))

                self.path_cm_micro = plot_confusion_matrix(self.input_object.cm_micro,
                                                           labels=['Positive', 'Negative'],
                                                           output_path=osp.join(self.output_path, 'cm_micro.png'))

            if self.input_object.get_calibration_curves:
                self.path_calibration_curves = plot_calibration_curves(prob_true=self.input_object.dict_prob_true,
                                                                       prob_pred=self.input_object.dict_prob_pred,
                                                                       hist_counts=self.input_object.dict_hist_counts,
                                                                       labels=self.input_object.class_labels,
                                                                       nbr_class=self.input_object.nbr_class,
                                                                       output_path=osp.join(self.output_path,
                                                                                            'calibration_curves.png'),
                                                                       bins=self.input_object.bins,
                                                                       bins_xticks=self.input_object.bins_xticks)

            if self.input_object.get_ROC_PR_curves:
                self.roc_pr_classes = plot_roc_pr_curves(data=self.input_object.vect_classes,
                                                         labels=self.input_object.class_labels,
                                                         nbr_class=self.input_object.nbr_class,
                                                         output_path=osp.join(self.output_path, 'roc_pr_curves.png'),
                                                         decimals=self.input_object.decimals)

            if self.input_object.get_hists_per_metrics:
                self.path_hists = self.input_object.plot_dataset_metrics_histograms()

    def to_json(self):
        """Create a report in the json format.
        """
        dict_export = self.input_object.dict_export
        json_object = json.dumps(dict_export, indent=4)
        with open(osp.join(self.output_path, 'report_metrics.json'), "w") as output_file:
            output_file.write(json_object)

    def to_md(self):
        """Create a report in the markdown format.
        """
        md_main = f"""
# ODEON - Metrics

## Micro Strategy

### Metrics
{self.df_to_md(self.round_df_values(self.input_object.df_report_micro, to_percent=True))}

(*) micro-F1 = micro-precision = micro-recall = accuracy

### Confusion matrix
![Confusion matrix micro](./{osp.basename(self.path_cm_micro)})

## Macro Strategy

### Metrics
{self.df_to_md(self.round_df_values(self.input_object.df_report_macro, to_percent=True))}

### Confusion matrix
![Confusion matrix macro](./{osp.basename(self.path_cm_macro)})

## Per Class Strategy

### Metrics
{self.df_to_md(self.round_df_values(self.input_object.df_report_classes, to_percent=True))}

"""
        md_elements = [md_main]

        if self.input_object.get_ROC_PR_curves:
            roc_pr_curves = f"""
## Roc PR Curves
![Roc PR curves](./{osp.basename(self.roc_pr_classes)})
"""
            md_elements.append(roc_pr_curves)

        if self.input_object.get_calibration_curves:
            calibration_curves = f"""
## Calibration Curves
![Calibration curves](./{osp.basename(self.path_calibration_curves)})
"""
            md_elements.append(calibration_curves)

        if self.input_object.get_hists_per_metrics:
            metrics_histograms = f"""
## Metrics Histograms

### Micro strategy
![Histograms micro](./{osp.basename(self.path_hists['micro'])})

### Macro strategy
![Histograms macro](./{osp.basename(self.path_hists['macro'])})

### Mean metrics
![Histograms means](./{osp.basename(self.path_hists['means'])})

### Per class strategy
"""
        class_histograms = []
        for class_name in self.input_object.class_labels:
            class_html = f"""
#### {class_name.capitalize()}:
![Histograms {class_name}](./{osp.basename(self.path_hists[class_name])})"""
            class_histograms.append(class_html)

        md_elements.append(metrics_histograms + "\n".join(class_histograms))

        with open(osp.join(self.output_path, 'multiclass_metrics.md'), "w") as output_file:
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

        main_html = """
            <h1><center> ODEON  Metrics</center></h1>
            """
        html_elements = [header_html, self.begin_html, main_html]

        micro_html = f"""
            <h2>Micro Strategy</h2>
            <h3>* Metrics</h3>
            {self.df_to_html(self.round_df_values(self.input_object.df_report_micro, to_percent=True))}
            (*) micro-F1 = micro-precision = micro-recall = accuracy

            <h3>* Confusion Matrix</h3>
            <p><img alt="Micro Confusion Matrix" src=./{osp.basename(self.path_cm_micro)} /></p>
            """
        html_elements.append(micro_html)

        if self.input_object.weighted:
            weigths_html = f'<p>Confusion matrix micro made with weights : {self.input_object.weights}</p>'
            html_elements.append(weigths_html)

        macro_html = f"""
            <h2>Macro Strategy</h2>
            <h3>* Metrics</h3>
            {self.df_to_html(self.round_df_values(self.input_object.df_report_macro, to_percent=True))}
            <h3>* Confusion Matrix</h3>
            <p><img alt="Macro Confusion Matrix" src=./{osp.basename(self.path_cm_macro)} /></p>
            """
        html_elements.append(macro_html)

        classes_html = f"""
            <h2>Per class Strategy</h2>
            <h3>* Metrics</h3>
            {self.df_to_html(self.round_df_values(self.input_object.df_report_classes, to_percent=True))}
            """
        html_elements.append(classes_html)

        if self.input_object.get_ROC_PR_curves:
            roc_pr_curves = f"""
            <h3>* ROC and PR Curves</h3>
            <p><img alt="ROC and PR Curves" src=./{osp.basename(self.roc_pr_classes)} /></p>
            """
            html_elements.append(roc_pr_curves)

        if self.input_object.get_calibration_curves:
            calibration_curves = f"""
            <h3>* Calibration Curves</h3>
            <p><img alt="Calibration Curve" src=./{osp.basename(self.path_calibration_curves)} /></p>
            """
            html_elements.append(calibration_curves)

        if self.input_object.get_hists_per_metrics:
            hists_class_html = []
            for class_name in self.input_object.class_labels:
                class_html = f"""
                <h4>{class_name.capitalize()} :</h4>
                <p><img alt="Histograms {class_name}" src=./{osp.basename(self.path_hists[class_name])} /></p>
                """
                hists_class_html.append(class_html)

            metrics_histograms = f"""
            <h2>Metrics Histograms</h2>

            <h3>Micro strategy</h3>
            <p><img alt="Histograms Micro" src=./{osp.basename(self.path_hists['micro'])} /></p>

            <h3>Macro strategy</h3>
            <p><img alt="Histograms Macro" src=./{osp.basename(self.path_hists['macro'])} /></p>

            <h3>Mean metrics</h3>
            <p><img alt="Histograms Means" src=./{osp.basename(self.path_hists['means'])} /></p>

            <h3>Per class strategy</h3>

            """ \
            + "\n".join(hists_class_html)

            html_elements.append(metrics_histograms)

        html_elements.append(end_html)

        with open(osp.join(self.output_path, 'metrics_multiclass.html'), "w") as output_file:
            for html_part in html_elements:
                output_file.write(html_part)
