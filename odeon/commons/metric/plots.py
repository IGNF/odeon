import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from cycler import cycler


def heatmap(data, row_labels, col_labels, axes=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Code from : https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data:
        A 2D numpy array of shape (N, M).
    row_labels:
        A list or array of length N with the labels for the rows.
    col_labels:
        A list or array of length M with the labels for the columns.
    axes:
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one. Optional.
    cbar_kw:
        A dictionary with arguments to `matplotlib.Figure.colorbar`. Optional.
    cbarlabel:
        The label for the colorbar.  Optional.
    **kwargs:
        All other arguments are forwarded to `imshow`.
    """
    if cbar_kw is None:
        cbar_kw = {}

    if not axes:
        axes = plt.gca()

    # Plot the heatmap
    image = axes.imshow(data, **kwargs)

    # Create colorbar
    cbar = axes.figure.colorbar(image, ax=axes, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    axes.set_xticks(np.arange(data.shape[1]))
    axes.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    axes.set_xticklabels(col_labels)
    axes.set_yticklabels(row_labels)
    axes.set_ylabel('Actual Class')
    axes.set_xlabel('Predicted Class')
    axes.xaxis.set_label_position('top')

    # Let the horizontal axes labeling appear on top.
    axes.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(axes.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in axes.spines.items():
        spine.set_visible(False)

    axes.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    axes.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    axes.grid(which="minor", color="w", linestyle='-', linewidth=3)
    axes.tick_params(which="minor", bottom=False, left=False)

    return image, cbar


def annotate_heatmap(image, data=None, valfmt="{x:.3f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    image
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = image.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = image.norm(threshold)
    else:
        threshold = image.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kwargs = dict(horizontalalignment="center", verticalalignment="center")
    kwargs.update(textkw)

    # # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    if isinstance(valfmt, (np.ndarray, np.generic)):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kwargs.update(color=textcolors[int(image.norm(data[i, j]) > threshold)])
                if valfmt[i, j] == 'nodata':
                    text = image.axes.text(j, i, valfmt[i, j], **kwargs)
                else:
                    decimals = 1 if data[i, j] >= 1 else 3
                    text = image.axes.text(j, i,
                                           str(np.round(data[i, j] / valfmt[i, j][0] if data[i, j] != 0 else 0,
                                                        decimals)) + valfmt[i, j][1], **kwargs)
                texts.append(text)
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kwargs.update(color=textcolors[int(image.norm(data[i, j]) > threshold)])
                text = image.axes.text(j, i, valfmt(data[i, j], None), **kwargs)
                texts.append(text)
    return texts


def get_cm_val_fmt(conf_mat, mark_no_data=False):
    """
    Function allowing to obtain a matrix containing the elements necessary to format each cell of a confusion matrix
    so that the number of observations can be entered in a cell. Each element of the matrix consist of tuple with a
    number to divide the value of the cm case and character to show the unit.
    ex: cm value = 3000 -> fmt (1000, 'k') -> '3k'.

    Parameters
    ----------
    cm : np.array
        Confusion matrix with float values to format.

    Returns
    -------
    np.array
        Matrix with elements to format the cm.
    """

    def find_val_fmt(value):
        """Return format element for one value.

        Parameters
        ----------
        value : float
            value to transform.

        Returns
        -------
        Tuple(int, str)
            Value to divide the input value, character to know in which unit is the input value.
        """
        length_dict = {0: (10**0, ''),
                       3: (10**3, 'k'),
                       6: (10**6, 'm'),
                       9: (10**9, 'g'),
                       12: (10**12, 't'),
                       15: (10**15, 'p')}
        divider, unit_char = None, None
        for i, length in enumerate(length_dict):
            number = str(value).split('.')[0]
            if len(number) < length + 1:
                divider = length_dict[list(length_dict)[i - 1]][0]
                unit_char = length_dict[list(length_dict)[i - 1]][1]
                break
            elif len(number) == length + 1:
                divider = length_dict[length][0]
                unit_char = length_dict[length][1]
                break
            elif i == len(length_dict) - 1:
                divider = length_dict[list(length_dict)[i]][0]
                unit_char = length_dict[list(length_dict)[i]][1]
        return (divider, unit_char)

    cm_val_fmt = np.zeros_like(conf_mat, dtype=object)
    for i in range(conf_mat.shape[0]):
        if mark_no_data and all(np.equal(conf_mat[i], 0)):
            cm_val_fmt[i] = ['nodata' for _ in range(conf_mat.shape[1])]
        else:
            for j in range(conf_mat.shape[1]):
                cm_val_fmt[i, j] = find_val_fmt(conf_mat[i, j])
    return cm_val_fmt


def plot_confusion_matrix(conf_mat, labels, output_path=None, per_class_norm=False, style=None, cmap="YlGn", figsize=None):
    """ Plot a confusion matrix with the number of observation in the whole input dataset.

    Parameters
    ----------
    conf_mat : np.array
        Confusion matrix.
    labels : list of str
        Labels for each class.
    output_path : str,
        Path of the output file
    cmap : str, optional
        colors to use in the plot, by default "YlGn"

    Returns
    -------
    str
        Ouput path of the image containing the plot.
    """
    if style is not None:
        plt.style.use(style)

    if figsize is None:
        if conf_mat.shape[0] < 10:
            figsize = (10, 7)
        elif conf_mat.shape[0] >= 10 and conf_mat.shape[0] <= 16:
            figsize = (12, 9)
        else:
            figsize = (16, 11)

    fig, axes = plt.subplots(figsize=figsize)
    cbarlabel = 'Coefficients values'

    if per_class_norm:
        dividend = conf_mat.astype('float')
        divisor = conf_mat.sum(axis=1)[:, np.newaxis]
        conf_mat = np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor != 0)

    image, _ = heatmap(conf_mat, labels, labels, axes=axes, cmap=cmap, cbarlabel=cbarlabel)
    # Rewrite cm with strings in order to fit the values into the figure.

    cm_val_fmt = get_cm_val_fmt(conf_mat)
    annotate_heatmap(image, valfmt=cm_val_fmt)

    fig.tight_layout(pad=3)

    if output_path is None:
        return fig
    else:
        plt.savefig(output_path)
        plt.close()
        return output_path


def plot_norm_and_value_cms(conf_mat, labels, output_path=None, per_class_norm=True, style=None, cmap="YlGn"):
    """Plot a confusion matrix with the number of observation and also another one with values
    normalized (per class or by the whole cm).

    Parameters
    ----------
    conf_mat : np.array
        Confusion matrix.
    labels : list of str
        Labels for each class.
    name_plot : str, optional
        Name of the output file, by default 'confusion_matrix.png'
    per_class_norm : bool, optional
        normalize per class or by the whole values in the cm, by default True
    cmap : str, optional
        colors to use in the plot, by default "YlGn"

    Returns
    -------
    str
        Ouput path of the image containing the plot.
    """
    if style is not None:
        plt.style.use(style)

    if conf_mat.shape[0] < 10:
        figsize = (20, 7)
    elif conf_mat.shape[0] >= 10 and conf_mat.shape[0] <= 16:
        figsize = (23, 9)
    else:
        figsize = (26, 11)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    cbarlabel = 'Coefficients values'
    fontsize = 12
    # On ax0, normalize cm
    dividend = conf_mat.astype('float')
    if not per_class_norm:
        divisor = np.sum(conf_mat.flatten())
    else:
        divisor = conf_mat.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor != 0)

    im0, _ = heatmap(cm_norm, labels, labels, axes=axs[1], cmap=cmap, cbarlabel=cbarlabel)
    cm_val_fmt_norm = get_cm_val_fmt(cm_norm, mark_no_data=True)
    annotate_heatmap(im0, data=np.round(cm_norm, decimals=3), valfmt=cm_val_fmt_norm)
    if not per_class_norm:
        axs[1].set_title('Normalized values', y=-0.1, pad=-14, fontsize=fontsize)
    else:
        axs[1].set_title('Normalized per actual class values', y=-0.1, pad=-14, fontsize=fontsize)

    im1, _ = heatmap(conf_mat, labels, labels, axes=axs[0], cmap=cmap, cbarlabel=cbarlabel)
    cm_val_fmt = get_cm_val_fmt(conf_mat)
    annotate_heatmap(im1, valfmt=cm_val_fmt)
    axs[0].set_title('Number of observations', y=-0.1, pad=-14, fontsize=fontsize)

    fig.tight_layout(pad=2)

    if output_path is None:
        return fig
    else:
        plt.savefig(output_path)
        plt.close()
        return output_path


def prepare_data_roc_curve(data):
    fpr = np.array(data['FPR'])
    tpr = np.array(data['Recall'])
    fpr, tpr = fpr[::-1], tpr[::-1]
    fpr, tpr = np.insert(fpr, 0, 0), np.insert(tpr, 0, 0)
    fpr, tpr = np.append(fpr, 1), np.append(tpr, 1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def prepare_data_pr_curve(data):
    precision = np.array(data['Precision'])
    recall = np.array(data['Recall'])
    precision = np.array([1 if p == 0 and r == 0 else p for p, r in zip(precision, recall)])
    idx = np.argsort(recall)
    recall, precision = recall[idx], precision[idx]
    recall, precision = np.insert(recall, 0, 0), np.insert(precision, 0, 1)
    recall, precision = np.append(recall, 1), np.append(precision, 0)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc


def plot_roc_pr_curves(data, labels, nbr_class, output_path, decimals):
    """
    Plot (html/md output type) or export (json type) data on ROC and PR curves.
    For ROC curve, points (0, 0) and (1, 1) are added to data in order to have a curve
    which begin at the origin and finish in the top right of the image.
    Same thing for PR curve but with the points (0, 1) and (1, 0).

    Parameters
    ----------
    name_plot : str, optional
        Desired name to give to the output image, by default 'multiclass_roc_pr_curves.png'

    Returns
    -------
    str
        Output path where an image with the plot will be created.
    """
    cmap_colors = [plt.get_cmap('rainbow')(1. * i/nbr_class) for i in range(nbr_class)]
    colors = cycler(color=cmap_colors)
    if nbr_class == 1:
        labels = [labels]

    plt.figure(figsize=(16, 8))

    plt.subplot(121)
    for class_i, color in zip(labels, colors):
        fpr, tpr, roc_auc = prepare_data_roc_curve(data[class_i] if nbr_class > 1 else data)
        plt.plot(fpr,
                 tpr,
                 label=f'{class_i} AUC = {round(roc_auc * 100, decimals - 2)}',
                 color=color['color'] if nbr_class > 1 else 'tab:blue')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Roc Curves')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(122)
    for class_i, color in zip(labels, colors):
        precision, recall, pr_auc = prepare_data_pr_curve(data[class_i] if nbr_class > 1 else data)
        plt.plot(recall,
                 precision,
                 label=f'{class_i} AUC = {round(pr_auc * 100, decimals - 2)}',
                 color=color['color'] if nbr_class > 1 else 'tab:blue')
    plt.plot([1, 0], [0, 1], 'r--')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)

    plt.tight_layout(pad=3)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_calibration_curves(prob_true, prob_pred, hist_counts, labels, nbr_class, output_path, bins, bins_xticks):
    """
    Plot or export data on calibration curves.
    Plot for output type html and md, export the data in a dict for json.

    Parameters
    ----------
    name_plot : str, optional
        Name to give to the output plot, by default 'multiclass_calibration_curves.png'

    Returns
    -------
    str
        Output path where an image with the plot will be created.
    """
    cmap_colors = [plt.get_cmap('rainbow')(1. * i/nbr_class) for i in range(nbr_class)]
    colors = cycler(color=cmap_colors)
    if nbr_class == 1:
        labels = [labels]
    plt.figure(figsize=(16, 8))
    # Plot 1: calibration curves
    plt.subplot(211)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for class_i, color in zip(labels, colors):
        plt.plot(prob_true[class_i] if nbr_class > 1 else prob_true,
                 prob_pred[class_i] if nbr_class > 1 else prob_pred,
                 "s-",
                 label=class_i,
                 color=color["color"] if nbr_class > 1 else 'tab:blue')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Calibration plots (reliability curve)')
    plt.ylabel('Fraction of positives')
    plt.xlabel('Probalities')

    # Plot 2: Hist of predictions distributions
    plt.subplot(212)
    for class_i, color in zip(labels, colors):
        plt.hist(bins[:-1],
                 weights=hist_counts[class_i] if nbr_class > 1 else hist_counts,
                 bins=bins,
                 histtype="step",
                 label=class_i,
                 lw=2,
                 color=color['color'] if nbr_class > 1 else 'tab:blue')
    plt.xticks(bins_xticks, bins_xticks)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Count')
    plt.xlabel('Mean predicted value')

    plt.tight_layout(pad=3)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_hists(hists, list_metrics, output_path, n_bins, bins_xticks, n_cols=3, size_col=5, size_row=4):
    """
    Plot metrics histograms.

    Parameters
    ----------
    list_metrics : list
        Name of the metrics to plot.
    n_cols : int, optional
        number of columns in the figure, by default 3
    size_col : int, optional
        size of a column in the figure, by default 5
    size_row : int, optional
        size of a row in the figure, by default 4
    name_plot : str, optional
        Name to give to the output plot, by default 'multiclass_hists_metrics.png'

    Returns
    -------
    str
        Output path where an image with the plot will be created.
    """
    # Here 7 corresponds to the average number of metrics.
    cmap_colors = [plt.get_cmap('turbo')(1. * i/7) for i in range(7)][1:]
    colors = cycler(color=cmap_colors)

    n_plot = len(list_metrics)
    n_rows = ((n_plot - 1) // n_cols) + 1
    plt.figure(figsize=(size_col * n_cols, size_row * n_rows))
    for i, plot_prop in enumerate(zip(list_metrics, colors)):
        metric, color = plot_prop[0], plot_prop[1]["color"]
        values = hists[metric]
        plt.subplot(n_rows, n_cols, i+1)
        plt.bar(range(len(values)), values, width=0.8, linewidth=2, capsize=20, color=color)
        if n_bins <= 20:
            plt.xticks(range(len(bins_xticks)), bins_xticks, rotation=-35)
        plt.title(f"{' '.join(metric.split('_'))}", fontsize=13)
        plt.xlabel("Values bins")
        plt.grid()
        plt.ylabel("Samples count")
    plt.tight_layout(pad=3)
    plt.savefig(output_path)
    plt.close()
    return output_path


def makegrid(images, preds, targets):
    assert len(images) == len(preds) == len(targets)
    return