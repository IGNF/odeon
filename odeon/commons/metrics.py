import numpy as np
import torch


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
    """
        Binarizes the detection masks
        Output is a mask with [n_classes, width, height] dimension with [0,1] values
        - for monoclass case, use of threshold to binarize
        - for multiclass case, use of argmax to binarize
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
    """
        Return the confusion matrix

        :param predictions: The predictions
        :param target: The target
        :param threshold: The threshold
        :return: The confusion matrix
    """
    mask = binarizes(predictions.cpu().numpy(), multilabel=multilabel)
    labels_masks = target.cpu().numpy()

    n_classes = mask.shape[1]

    cl = np.arange(n_classes)
    cms = np.zeros((2, 2), dtype=np.uint64)
    for c in np.nditer(cl):
        target = mask[:, c, :, :].flatten()
        prediction = labels_masks[:, c, :, :].flatten()
        cms = cms + get_binary_confusion_matrix(prediction, target)

    return cms

def get_binary_confusion_matrix(prediction, target):
    """
        Returns the confusion matrix for one class or threshold.

        TP (true positives): the number of correctly classified pixels (1 -> 1)
        TN (true negatives): the number of correctly not classified pixels (0 -> 0)
        FP (false positives): the number of pixels wrongly classified (0 -> 1)
        FN (false negatives): the number of pixels wrongly not classifed (1 -> 0)

        :param target: The binary ground truth
        :param prediction: The binary inference
        :return: The confusion matrix [[TP,FN],[FP,TN]]
    """

    tp = np.sum(np.logical_and(target, prediction))
    tn = np.sum(np.logical_not(np.logical_or(target, prediction)))
    fp = np.sum(prediction[target == 0] == 1)
    fn = np.sum(prediction[target == 1] == 0)

    return np.array([[tp, fn], [fp, tn]])

def get_iou_metrics(cm):
    """
        Returns the IOU metric for the provided confusion matrices

        Iou:
        The Intersection-Over-Union is a common evaluation metric for semantic image segmentation.
        iou = TP / (TP+FP+FN)

        :param cm: The confusion matrix [[TP,FN],[FP,TN]]
        :return: The IOU metrics
    """

    m = np.nan
    numerator = cm[0, 0]
    denominator = cm[0, 0] + cm[1, 0] + cm[0, 1]
    if denominator != 0:
        m = numerator / denominator

    return m
