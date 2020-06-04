import logging
import os

import torch

from nn.history import History

logger = logging.getLogger(__package__)

class TrainingEngine:
    """TrainingEngine class

        Parameters
        ----------
        model : :class:`nn.Module`
            model instance
        loss: :class:`nn.Module`
            loss instance
        optimizer: :class:`Optimizer`
            optimizer class
        lr_scheduler: object
            learning rate scheduler object
        output_folder: str
            output folder where model checkpoints and history file are written
        epochs : int, optional
            number of epochs, default 300
        output_filename: str
            output filename if omitted file name is generated from hyperparameters, default None
        """
    def __init__(self, model, loss, optimizer, lr_scheduler, output_folder, epochs=300, output_filename="model.pth",
                 **kwargs):
        """TrainingEngine class

        Parameters
        ----------
        model : :class:`nn.Module`
            model instance
        loss: :class:`nn.Module`
            loss instance
        optimizer: :class:`Optimizer`
            optimizer class
        lr_scheduler: object
            learning rate scheduler object
        output_folder: str
            output folder where model checkpoints and history file are written
        epochs : int, optional
            number of epochs, default 300
        output_filename: str
            output filename if omitted file name is generated from hyperparameters, default None
        """
        self.device = kwargs.get('device', 'cpu')
        self.net = model.cuda(self.device) if self.device.startswith('cuda') else model
        self.epochs = epochs
        self.batch_size = kwargs.get('batch_size', 16)
        self.patience = kwargs.get('batch_size', 20)

        self.save_history = kwargs.get('save_history', True)
        self.continue_training = kwargs.get('continue_training', False)

        self.optimizer = optimizer
        self.loss = loss.cuda(self.device) if self.device.startswith('cuda') else loss
        self.lr_scheduler = lr_scheduler

        self.output_folder = output_folder
        self.output_filename = output_filename

    def train(self, train_loader, val_loader):

        logger.info(f'''
        Start training:
            Epochs: {self.epochs}
            Batch size: {self.batch_size}
            Learning rate: {self.optimizer.param_groups[0]['lr']}
            Training size: {len(train_loader.dataset)}
            Validation size: {len(val_loader.dataset)}
            Save history: {self.save_history}
            Device: {self.device}
        ''')

        patience_counter = 0
        epoch_start = 0
        prec_val_loss = 1000

        # history
        base_history_file = os.path.join(self.output_folder, f'{os.path.splitext(self.output_filename)}')
        history = History(base_history_file, self.continue_training)

        model_filepath = os.path.join(self.output_folder, self.output_filename)

        # training loop
        for epoch in range(epoch_start, self.epochs):
            # switch to train mode
            self.net.train()

            # run a pass on current epoch
            train_loss, train_iou, avg_time = self._train_epoch(train_loader)

            # switch to evaluate mode
            self.net.eval()
            # run the validation pass
            with torch.no_grad():
                val_loss, val_iou = self._validate_epoch(val_loader)

            self.lr_scheduler.step(val_loss)

            logger.info(f"train_loss = {train_loss:03f}, val_loss = {val_loss:03f}, val_iou = {val_iou:03f}")

            # update history
            history.update(epoch, avg_time, train_loss, val_loss, train_iou, val_iou,
                           self.optimizer.param_groups[0]['lr'])

            # save model if val_loss has decreased
            if prec_val_loss > val_loss:
                logger.info(f"Saving {model_filepath}")
                torch.save(self.net.state_dict(), model_filepath)

                if self.save_history:
                    history.save()
                    history.plot()

                prec_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # stop training if patience is reached
            if patience_counter == self.patience:
                logger.info(f"Model has not improved since {self.patience} epochs, train stopped.")
                break

    def _train_epoch(self, loader):
        return 0.5, 0.75, 600

    def _validate_epoch(self, loader):
        return 0.5, 0.75
