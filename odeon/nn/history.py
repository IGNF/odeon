import os
import json
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

class History:
    """History class to manage metrics storage during training, export to json file and plots

    Parameters
    ----------
    base_path : str
        base path for json and plot files
    update : bool, optional
        update existing files, by default False
    train_iou : bool, optional
        store iou values during training, by default False
    """

    def __init__(self, base_path, update=False, train_iou=False):

        key_metrics = ['epoch', 'duration', 'train_loss', 'val_loss', 'val_mean_iou', 'learning_rate']
        if train_iou is True:
            key_metrics.append('train_mean_iou')
        self.history_dict = {
            key: [] for key in key_metrics
        }

        self.base_path = base_path

        if update is True and os.path.exists(self.history_file):
            with open(self.history_file, 'r') as file:
                self.history_dict = json.load(file)

    def get_current_epoch(self):
        return self.history_dict['epoch'][-1]

    def update(self, epoch, duration, train_loss, val_loss, learning_rate, val_mean_iou, train_mean_iou=None):
        """update history dict

        Parameters
        ----------
        epoch : int
        duration : float
        train_loss : float
        val_loss : float
        train_mean_iou : float
        val_mean_iou : float
        learning_rate : float
        """
        self.history_dict['epoch'].append(epoch)
        self.history_dict['duration'].append(duration)
        self.history_dict['train_loss'].append(train_loss)
        self.history_dict['val_loss'].append(val_loss)
        self.history_dict['val_mean_iou'].append(val_mean_iou)
        self.history_dict['learning_rate'].append(learning_rate)
        if train_mean_iou is not None:
            self.history_dict['train_mean_iou'].append(train_mean_iou)

    def save(self):
        history_file = f'{self.base_path}_history.json'
        with open(history_file, 'w') as file:
            json.dump(self.history_dict, file)

    def plot(self):
        train_loss = np.array(self.history_dict['train_loss'])
        val_loss = np.array(self.history_dict['val_loss'])
        val_miou = np.array(self.history_dict['val_mean_iou'])
        if 'train_mean_iou' in self.history_dict:
            train_miou = np.array(self.history_dict['train_mean_iou'])

        # Loss
        plt.figure(1)
        t, = plt.plot(train_loss, 'b')
        v, = plt.plot(val_loss, 'g')

        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend((t, v), ('train', 'validation'))

        plt.savefig(f'{self.base_path}_loss.png', bbox_inches='tight')

        plt.figure(2)

        if 'train_mean_iou' in self.history_dict:
            t, = plt.plot(train_miou, 'b')
        v, = plt.plot(val_miou, 'g')

        if 'train_mean_iou' in self.history_dict:
            t, = plt.plot(train_miou, 'b')
            plt.legend((t, v), ('train', 'validation'), loc='upper left')
        else:
            plt.legend((v,), ('validation',), loc='upper left')

        plt.title('Mean IoU')
        plt.xlabel('Epochs')
        plt.ylabel('miou')

        plt.savefig(f'{self.base_path}_miou.png', bbox_inches='tight')
