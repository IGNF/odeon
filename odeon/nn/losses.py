"""
Parts of this file inspired by
(solaris)[https://solaris.readthedocs.io/en/latest/_modules/solaris/nets/_torch_losses.html#lovasz_hinge_flat]
and work from SpaceNet challenges participants
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import filterfalse
import numpy as np
from odeon import LOGGER

COMBOLOSS_BCE = 0.75
COMBOLOSS_JACCARD = 0.25


def build_loss_function(loss_name, class_weight=None):
    if loss_name == "ce":
        if class_weight is not None:
            LOGGER.info(f"Weights used: {class_weight}")
            with torch.no_grad():
                class_weight = torch.FloatTensor(class_weight)
        return CrossEntropyWithLogitsLoss(weight=class_weight)
    elif loss_name == "bce":
        return BCEWithLogitsLoss()
    elif loss_name == "focal":
        return FocalLoss2d()
    elif loss_name == "combo":
        return ComboLoss({'bce': COMBOLOSS_BCE, 'jaccard': COMBOLOSS_JACCARD})


class BCEWithLogitsLoss(nn.Module):
    """Binary cross entropy loss with logits
    Loss is computed each class separately and reduced regarding reduction param
    Raw logits are flattened before invoking nn.BCEWithLogitsLoss which combines Sigmoid and BCELoss

    Parameters
    ----------
    weight : list of float, optional
        weights applied to loss computation, by default None
    reduction : str, optional
        reduction to apply to output, by default 'mean'
    pos_weight : list of float, optional
        , by default None
    """
    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight, reduction=reduction, pos_weight=pos_weight)

    def forward(self, logits, targets):
        # flatten should not be needed
        # probs_flat = logits.view(-1)  # Flatten
        # targets_flat = targets.view(-1)  # Flatten

        # we should have logits of shape N,1,H,W and target also of shape N,1,H,W
        # targets need to be converted to float for pytorch BCEWithLogitsLoss
        # see :
        # https://discuss.pytorch.org/t/multi-label-binary-classification-result-type-float-cant-be-cast-to-the-desired-output-type-long/117915
        return self.bce_loss(logits, targets.float())


class CrossEntropyWithLogitsLoss(nn.Module):
    """Cross entropy loss with logits
    Labels are flattened using argmax function and CrossEntropyLoss uses a LogSoftmax function.

    Parameters
    ----------
    weight : [type], optional
        [description], by default None
    reduction : str, optional
        reduction to apply to output, by default 'mean'
    """

    def __init__(self, weight=None, reduction='mean'):

        super(CrossEntropyWithLogitsLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight, reduction=reduction)

    def forward(self, logits, targets):
        # flatten masks to get rid of channel dimension
        # force/cast target tensor to long
        targets = torch.argmax(targets.long(), dim=1)
        return self.cross_entropy(logits, targets)


class ComboLoss(nn.Module):
    def __init__(self, weights, per_image=False):
        super().__init__()
        self.weights = weights
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss(per_image=False)
        self.jaccard = JaccardLoss(per_image=False)
        self.lovasz = LovaszLoss(per_image=per_image)
        self.focal = FocalLoss2d()
        self.mapping = {'bce': self.bce,
                        'dice': self.dice,
                        'focal': self.focal,
                        'jaccard': self.jaccard,
                        'lovasz': self.lovasz}
        # self.expect_sigmoid = {'dice', 'focal', 'jaccard', 'lovasz_sigmoid'}
        self.values = {}

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        # sigmoid_input = torch.sigmoid(outputs)
        for k, v in weights.items():

            if not v:

                continue

            # val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)
            val = self.mapping[k](outputs, targets)
            self.values[k] = val
            loss += self.weights[k] * val

        return loss


class SoftDiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


def dice_round(preds, trues):
    preds = preds.float()
    return soft_dice_loss(preds, trues)


def soft_dice_loss(outputs, targets, per_image=False):
    batch_size = outputs.size()[0]
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def jaccard(outputs, targets, per_image=True, non_empty=False, min_pixels=5):
    batch_size = outputs.size()[0]
    eps = 1e-3
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(dice_target, dim=1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
    if non_empty:
        assert per_image is True
        non_empty_images = 0
        sum_loss = 0
        for i in range(batch_size):
            if target_sum[i] > min_pixels:
                sum_loss += losses[i]
                non_empty_images += 1
        if non_empty_images == 0:
            return 0
        else:
            return sum_loss / non_empty_images
    return losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, logits, target):
        input = torch.sigmoid(logits)
        return soft_dice_loss(input, target, per_image=self.per_image)


class JaccardLoss(nn.Module):

    def __init__(
        self,
        weight=None,
        size_average=True,
        per_image=False,
        non_empty=False,
        apply_sigmoid=False,
        min_pixels=5
        ):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, input, target):
        input = torch.sigmoid(input)
        return jaccard(
            input, target, per_image=self.per_image,
            non_empty=self.non_empty, min_pixels=self.min_pixels)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts.float() - gt_sorted.float().cumsum(0)
    union = gts.float() + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):

    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss

    Arguments
    ---------
    logits: :class:`torch.Tensor`
        Logits at each prediction (between -inf and +inf)
    labels: :class:`torch.Tensor`
        binary ground truth labels (0 or 1)

    Returns
    -------
    loss : :class:`torch.Tensor`
        Lovasz loss value for the input logits and labels.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def mean(val, ignore_nan=False, empty=0):
    """nanmean compatible with generators.
    """

    l_iter = iter(val)
    if ignore_nan:
        l_iter = filterfalse(np.isnan, val)
    try:
        n = 1
        acc = next(val)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l_iter, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszLoss(nn.Module):

    def __init__(self, ignore_index=255, per_image=True):

        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, outputs, targets):

        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return lovasz_hinge(outputs, targets, per_image=self.per_image, ignore=self.ignore_index)


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, ignore_index=255):

        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):

        outputs = torch.sigmoid(logits)
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()
