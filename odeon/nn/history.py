import os
import json
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

class History:

    def __init__(self, base_path, update=False):
        """History class to manage metrics storage during training, export to json file and plots

        Parameters
        ----------
        base_path : str
            base path for json and plot files
        update : bool, optional
            update existing files, by default False
        """

        key_metrics = ['epoch', 'duration', 'train_loss', 'val_loss', 'train_mean_iou', 'val_mean_iou', 'learning_rate']
        self.history_dict = {
            key: [] for key in key_metrics
        }

        self.history_file = f'{base_path}_history.json'
        self.plot_loss_file = f'{base_path}_loss.png'
        self.plot_miou_file = f'{base_path}_miou.png'

        if update is True and os.path.exists(self.history_file):
            with open(self.history_file, 'r') as file:
                self.history_dict = json.load(file)

    def get_current_epoch(self):
        return self.history_dict['epoch'][-1]

    def update(self, epoch, duration, train_loss, val_loss, train_mean_iou, val_mean_iou, learning_rate):
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
        self.history_dict['train_mean_iou'].append(train_mean_iou)
        self.history_dict['val_mean_iou'].append(val_mean_iou)
        self.history_dict['learning_rate'].append(learning_rate)

    def save(self):
        with open(self.history_file, 'w') as file:
            json.dump(self.history_dict, file)

    def plot(self):
        train_loss = np.array(self.history_dict['train_loss'])
        val_loss = np.array(self.history_dict['val_loss'])
        train_miou = np.array(self.history_dict['train_mean_iou'])
        val_miou = np.array(self.history_dict['val_mean_iou'])

        # Loss
        plt.figure(1)
        t, = plt.plot(train_loss, 'b')
        v, = plt.plot(val_loss, 'g')

        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend((t, v), ('train', 'validation'))

        plt.savefig(self.plot_loss_file, bbox_inches='tight')

        # Accuracy
        plt.figure(2)
        t, = plt.plot(train_miou, 'b')
        v, = plt.plot(val_miou, 'g')

        plt.title('Mean IoU')
        plt.xlabel('Epochs')
        plt.ylabel('miou')
        plt.legend((t, v), ('train', 'validation'), loc='upper left')

        plt.savefig(self.plot_miou_file, bbox_inches='tight')
