""" Metrics Utils"""
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from odeon import LOGGER


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

    def get_avg(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def plot_conf_matrix():

    pass


def plot_image_debug_validation_loop(img, target, pred, color_map, output_file, fig_size: Tuple[int] = (20, 20),
                                     subtitle: str = "", fontsize=16, patches: Optional[List[mpatches.Patch]] = None):

    color_map = np.array(color_map)
    img = img.astype("uint8")
    pred = pred.astype("uint8")
    target = target.astype("uint8")
    img_h, img_w, img_c = img.shape
    # pred_w, pred_h, pred_c = pred.shape
    # target_w, target_h, target_c = target.shape
    rgb_pred = color_map[pred.astype(int)].astype("uint8")
    rgb_target = color_map[target.astype(int)].astype("uint8")
    # LOGGER.info(f"shape rgb_pred {rgb_pred.shape}")
    # LOGGER.info(f"shape img {img.shape}")
    # LOGGER.info(f"shape rgb_trget {rgb_target.shape}")
    rgb_error = (np.ones((img_w, img_h, img_c)) * 255).astype("uint8")
    target_r = np.repeat(target[:, :, np.newaxis], 3, axis=2)
    pred_r = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
    rgb_error = np.where(target_r != pred_r, rgb_error, img).astype("uint8")
    rgb_target_error = np.where(target_r != pred_r, rgb_target, img).astype("uint8")
    rgb_pred_error = np.where(target_r != pred_r, rgb_pred, img).astype("uint8")
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(subtitle)
    if patches is not None:
        fig.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1, handles=patches)

    ax = []
    ax.append(fig.add_subplot(3, 3, 1))
    ax[-1].clear()
    ax[-1].set_title("image")
    plt.imshow(img)
    # axarr[0, 0].set_title("img 1")

    ax.append(fig.add_subplot(3, 3, 2))
    ax[-1].clear()
    ax[-1].set_title("image with errors")
    plt.imshow(rgb_error)

    ax.append(fig.add_subplot(3, 3, 3))
    ax[-1].clear()
    ax[-1].set_title("image with target label on errors")
    plt.imshow(rgb_target_error)
    # axarr[0, 0].set_title("img 1")

    ax.append(fig.add_subplot(3, 3, 4))
    ax[-1].clear()
    ax[-1].set_title("image with prediction label on errors")

    plt.imshow(rgb_pred_error)

    ax.append(fig.add_subplot(3, 3, 5))
    ax[-1].clear()
    ax[-1].set_title("image with prediction label")
    plt.imshow(rgb_pred)

    ax.append(fig.add_subplot(3, 3, 6))
    ax[-1].clear()
    ax[-1].set_title("image with target label")
    plt.imshow(rgb_target)

    # axarr[0, 1].set_title("img 2")
    # axarr[1, 1].set_title("sub img 2")
    plt.savefig(output_file)
    plt.close(fig)
