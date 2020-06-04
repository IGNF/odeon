import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_history(history, filename):
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    train_miou = np.array(history['train_mean_iou'])
    val_miou = np.array(history['val_mean_iou'])

    # Loss
    plt.figure(1)
    t, = plt.plot(train_loss, 'b')
    v, = plt.plot(val_loss, 'g')

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend((t, v), ('train', 'validation'))

    plt.savefig(filename + "_loss.png", bbox_inches='tight')

    # Accuracy
    plt.figure(2)
    t, = plt.plot(train_miou, 'b')
    v, = plt.plot(val_miou, 'g')

    plt.title('Mean IoU')
    plt.xlabel('Epochs')
    plt.ylabel('miou')
    plt.legend((t, v), ('train', 'validation'), loc='upper left')

    plt.savefig(filename + "_miou.png", bbox_inches='tight')
