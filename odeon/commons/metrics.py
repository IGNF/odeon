import numpy as np
import torch
import torch.nn.functional as F
from odeon.commons.exception import OdeonError, ErrorCodes
# from odeon import LOGGER


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def binarizes(detection, threshold=0.5, multilabel=False):
    """Binarizes the detection masks
       Output is a mask with [n_classes, width, height] dimension with [0,1] values
        - for monoclass case, use of threshold to binarize
        - for multiclass case, use of argmax to binarize

    Parameters
    ----------
    detection : ndarray
        detection mask
    threshold : float, optional
        threshold, by default 0.5
    multilabel : bool, optional
        activate multilabel mode, by default False

    Returns
    -------
    ndarray
        binarized mask
    """
    no_of_class = detection.shape[1]
    if no_of_class == 1 or multilabel:  # Monoclass or multilabel
        assert threshold is not None
        tmp = detection.copy()
        tmp[detection > threshold] = 1
        tmp[detection <= threshold] = 0
        return tmp.copy()
    else:  # Multiclass monolabel

        labels = np.argmax(detection, axis=1)
        cl = np.arange(no_of_class)
        v_other = no_of_class + 1
        result = []

        for c in np.nditer(cl):

            tmp = labels.copy()
            tmp[tmp != c] = v_other
            tmp[tmp == c] = 1
            tmp[tmp == v_other] = 0
            result.append(tmp.copy())

        result = np.array(result)
        result = result.swapaxes(0, 1)
        return result


def get_confusion_matrix(predictions, target, multilabel=False):
    """Return the confusion matrix

    The confusion matrix is a numpy array of size 2*2 of form
                     TP | FN
                     -------
                     FP | TN
    The muticlass confusion matrix is the sum of the binary confusion
    matrix for each class.

    Parameters
    ----------
    predictions : ndarray
        predictions
    target : ndarray
        labels
    multilabel : bool, optional
        activate multilabel mode, by default False

    Returns
    -------
    ndarray
        confusion matrix
    """

    mask = binarizes(predictions.cpu().numpy(), multilabel=multilabel)
    n_classes = mask.shape[1]

    # if multilabel case target mask could have common pixel
    # if not multilabel we force target to have only one class by applying an argmax
    # and then get back label as one hot encoding of dim N C W H
    if multilabel:
        labels_masks = target.cpu().numpy()
    else:
        labels_masks = F.one_hot(torch.argmax(target, axis=1), num_classes=n_classes)
        # one hot encoding use last dim so we use transpose to convert
        # from (N,W,H,C) to (N,C,W,H) tensor
        labels_masks = labels_masks.permute(0, 3, 1, 2).cpu().numpy()

    cl = np.arange(n_classes)
    cms = np.zeros((2, 2), dtype=np.uint64)
    for c in np.nditer(cl):
        target = mask[:, c, :, :].flatten()
        prediction = labels_masks[:, c, :, :].flatten()
        cms = cms + get_binary_confusion_matrix(prediction, target)

    return cms


def get_confusion_matrix_torch(predictions, target, multilabel=False, cuda=False, threshold=0.5):
    """Return the confusion matrix

    The confusion matrix is :

      * a torch tensor of size C*C with C number of class
        (fom prediction shape) in multiclass case (no multilabel)
      * a torch tensor of size C*2*2 if multilabel


    Parameters
    ----------
    predictions : torch tensor of dim N,C,W,H (float)
        predictions
    target : torch tensor of dim N,C,W,H (long)
        labels
    multilabel : bool, optional
        activate multilabel mode, by default False
    cude : bool, optional
        activate computation on cuda (GPU) if True, else use torch on CPU

    Returns
    -------
    ndarray
        confusion matrix
    """
    # first we obtain number of class from prediction shape
    num_class = predictions.shape[1]
    # need to detach if we are in training mode
    pred_detach = predictions.detach()
    target_detach = target.detach()
    if not multilabel:
        # predictions and target are transformed from one-hot to int with argmax
        preds_cm = pred_detach.argmax(1).view(-1)
        target_cm = target_detach.argmax(1).view(-1)  # so no multilabel here
        # the trick to speed up torch computation of confusion matrix is to
        # transform the problem to only use optimized functions of torch (GPU/ or
        # parallelize on cpu).
        # Here we use a linear tranformation and an histogramm computation.
        # Example for 3 classes cases :
        #  first encore each possible pred/target values to an unique value using base 3
        #  unique_value = 3^1 * pred + 3^0 * target
        #
        #       0     |    1   |   2                    0  |  1  |   2
        #      ------------------------              -------------------
        #   0 | 0*3+0 | 0*3+1  | 0*3+1             1 | 0   |  1  |  2
        #   1 | 1*3+0 | 1*3+1  | 1*3+2      ->     2 | 3   |  4  |  5
        #   2 | 2*3+0 | 2*3+1  | 2*3+2             3 | 6   |  7  |  8
        #
        #  then compute histogram of unique values with bincount
        #            0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8
        #   hist = ------------------------------------------------------
        #           95  |  2  |  3  |  8  |  85 |  7  |  4  |  5  |  91
        #
        # and finally reshape histogramm to 2D
        #
        #               0  |  1  |  2
        #            -------------------
        #          0 |  95  |  2  |  3
        # cm =     1 |   8  | 85  |  7
        #          2 |   4  |  5  | 91
        #
        # transform of (pred, target) couple in unique value using a base N
        # encoding
        y = num_class * preds_cm + target_cm
        # compute histogramm of each possible unique value (each possible couple pred/target)
        y = torch.bincount(y)
        # as output of bincout is of len max(y) if not all class exists in current
        # batch we need to pad with zero values (by concatenation)
        if len(y) < num_class * num_class:
            y_comp = torch.zeros(num_class * num_class - len(y), dtype=torch.long)
            if cuda:
                y_comp = y_comp.cuda()
            y = torch.cat([y, y_comp])
        # finally we reshape 1D array to 2D confusion matrix
        y = y.reshape(num_class, num_class)
    else:

        # vectorize binary matrix computation by using tensor with
        # first dim = C (class)
        target_reshaped = target_detach.transpose(0, 1).reshape(num_class, -1).long()
        pred_reshaped = pred_detach.transpose(0, 1).reshape(num_class, -1)
        pred_reshaped = (pred_reshaped > threshold).long()

        target_total = target_reshaped.sum(dim=1)
        pred_total = pred_reshaped.sum(dim=1)

        tp = (target_reshaped * pred_reshaped).sum(dim=1)
        fp = pred_total - tp
        fn = target_total - tp
        tn = target_reshaped.shape[1] - tp - fp - fn

        # reshape result as num_class, 2, 2 tensor
        y = torch.stack([tp, fp, fn, tn], dim=1).reshape(-1, 2, 2)
        if cuda:
            y.cuda()

    return y


def get_binary_confusion_matrix(prediction, target):
    """Returns the confusion matrix for one class or threshold.
       TP (true positives): the number of correctly classified pixels (1 -> 1)
       TN (true negatives): the number of correctly not classified pixels (0 -> 0)
       FP (false positives): the number of pixels wrongly classified (0 -> 1)
       FN (false negatives): the number of pixels wrongly not classifed (1 -> 0)

    Parameters
    ----------
    prediction : ndarray
        binary inference
    target : ndarray
        binary ground truth

    Returns
    -------
    ndarray
        confusion matrix [[TP,FN],[FP,TN]]
    """

    tp = np.sum(np.logical_and(target, prediction))
    tn = np.sum(np.logical_not(np.logical_or(target, prediction)))
    fp = np.sum(prediction[target == 0] == 1)
    fn = np.sum(prediction[target == 1] == 0)

    return np.array([[tp, fn], [fp, tn]])


def get_iou_metrics(cm):
    """Returns the IOU metric for the provided confusion matrices

    IoU: The Intersection-Over-Union is a common evaluation metric for semantic image segmentation.
        iou = TP / (TP+FP+FN)

    Parameters
    ----------
    cm : ndarray
        confusion matrix [[TP,FN],[FP,TN]]

    Returns
    -------
    float
       Miou

    """
    # LOGGER.info(f"cm = {cm}")
    m = np.nan
    numerator = cm[0, 0]
    denominator = cm[0, 0] + cm[1, 0] + cm[0, 1]
    if denominator != 0:
        m = numerator / denominator

    return m


def get_iou_metrics_torch(cm, micro=True, cuda=False):
    """Returns the IOU metric for the provided confusion matrices

    IoU: The Intersection-Over-Union is a common evaluation metric for semantic image segmentation.
        iou = TP / (TP+FP+FN)

    If "micro" the IoU is computed globally (sum of binary confusion matrix for each class)
    Else we use a macro IoU computed by mean of all the class IoU

    Parameters
    ----------
    cm : Tensor
        confusion matrix of size N*2*2 (multilabel) or N*N
    type : String
        "micro" or "macro".

    Returns
    -------
    float
        IoU metric

    """
    if cuda:
        cm_array = cm.cpu().detach().numpy()
    else:
        cm_array = cm.numpy()

    m = np.nan
    num_dim = len(cm.size())
    # LOGGER.info(f"num_dim = {num_dim:03f}")
    if num_dim == 2:
        # with confusion matrix cm
        # for each class i
        #   TP_i = cm [i, i]
        #   FN_i = sum_j cm[i , j] - cm [i, i]  (sum on dim 1)
        #   FP_i = sum_j cm[j , i] - cm [i, i]  (sum on dim 0)
        if micro:
            # we use nansum to handle np.nan values inside cm
            cm_00 = np.nansum(np.diag(cm_array))  # TP = sum of cm diag element without nan
            cm_10 = np.nansum(cm_array.sum(0)-np.diag(cm_array))  # FP_all = sum FP_i
            cm_01 = np.nansum(cm_array.sum(1)-np.diag(cm_array))  # FN_all = sum FN_i
            if cm_00 != 0:
                m = cm_00 / (cm_00 + cm_10 + cm_01)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                # we could ignore divide by zero and np.nan result as is managed by next compute
                # ious is an 1D array of class iou
                # for each class :
                #   Inter_i = TP_i
                #   Union_i = TP_i + FP_i + FN_i = sum_j cm[i , j] +  sum_j cm[i , j] - cm [i, i]
                ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
            # do not count classes that are not present in the dataset in the mean IoU
            m = np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()

    elif num_dim == 3:
        if micro:
            # sum all class binary confusion matrix and then compute iou
            cm_all = np.nansum(cm_array, axis=0)
            m = get_iou_metrics(cm_all)
        else:
            # compute iou by class and then take the mean
            # from N * 2 *2 t* 2*2*N
            cm_array = cm_array.transpose(1, 2, 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ious = cm_array[0, 0] / (cm_array[0, 0] + cm_array[1, 0] + cm_array[0, 1])
            m = np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()

    else:
        raise OdeonError(
            message=f"confusion matrix of from size {cm.size()}, should be of type N*N or N*2*2",
            error_code=ErrorCodes.ERR_TRAINING_ERROR)

    return m.astype(float)
