import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

import torch

from odeon.nn.history import History
from odeon.commons.metrics import AverageMeter, get_confusion_matrix, get_iou_metrics

from odeon import LOGGER

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
        verbose: boolean, optional
            verbosity adds iou computation during training, default False
        """

    def __init__(self, model, loss, optimizer, lr_scheduler, output_folder, output_filename="model.pth",
                 verbose=False, **kwargs):

        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = model.cuda(self.device) if self.device.startswith('cuda') else model
        self.epochs = kwargs.get('epochs')
        self.batch_size = kwargs.get('batch_size')
        self.patience = kwargs.get('patience')

        self.save_history = kwargs.get('save_history')
        self.continue_training = kwargs.get('continue_training')

        self.optimizer = optimizer
        self.loss = loss.cuda(self.device) if self.device.startswith('cuda') else loss
        self.lr_scheduler = lr_scheduler

        self.output_folder = output_folder
        self.output_filename = output_filename

        self.train_iou = verbose

    def train(self, train_loader, val_loader):

        LOGGER.info(f'''Training:
            Model: {type(self.net).__name__}
            Batch size: {self.batch_size}
            Loss function: {type(self.loss).__name__}
            Optimizer: {type(self.optimizer).__name__}
            Learning rate: {type(self.lr_scheduler).__name__} starting at {self.optimizer.param_groups[0]['lr']}
            Save history: {self.save_history}
            Device: {self.device}
        ''')

        patience_counter = 0
        epoch_start = 0
        prec_val_loss = 1000

        # history
        base_history_file = os.path.join(self.output_folder, f'{os.path.splitext(self.output_filename)[0]}')
        history = History(base_history_file, update=self.continue_training, train_iou=self.train_iou)

        model_filepath = os.path.join(self.output_folder, self.output_filename)

        # training loop
        for epoch in range(epoch_start, self.epochs):
            self.epoch_counter = epoch
            # switch to train mode
            self.net.train()

            # run a pass on current epoch
            train_loss, train_miou, avg_time = self._train_epoch(train_loader)

            # switch to evaluate mode
            self.net.eval()
            # run the validation pass
            with torch.no_grad():
                val_loss, val_miou = self._validate_epoch(val_loader)

            self.lr_scheduler.step(val_loss)

            LOGGER.info(f"train_loss = {train_loss:03f}, val_loss = {val_loss:03f}")
            if self.train_iou:
                LOGGER.info(f"train_miou = {train_miou:03f}, val_miou = {val_miou:03f}")
            else:
                LOGGER.info(f"val_miou = {val_miou:03f}")

            # update history
            history.update(epoch, avg_time, train_loss, val_loss,
                           self.optimizer.param_groups[0]['lr'], val_miou, train_mean_iou=train_miou)

            # save model if val_loss has decreased
            if prec_val_loss > val_loss:
                LOGGER.info(f"Saving {model_filepath}")
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
                LOGGER.info(f"Model has not improved since {self.patience} epochs, train stopped.")
                break

    def _train_epoch(self, loader):

        losses = AverageMeter("train_loss")
        confusion_matrix = np.zeros((2, 2), dtype=np.uint64)

        with tqdm(total=len(loader),
                  desc=f"Epochs {self.epoch_counter + 1}/{self.epochs}",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]') as pbar:

            for sample in loader:

                images = sample['image'].cuda(self.device) if self.device.startswith('cuda') else sample['image']
                masks = sample['mask'].cuda(self.device) if self.device.startswith('cuda') else sample['mask']

                # clear gradient
                self.optimizer.zero_grad()

                # forward pass
                logits = self.net(images)

                # predictions
                if self.net.n_classes == 1:
                    preds = torch.sigmoid(logits)
                else:
                    preds = torch.softmax(logits, dim=1)

                # compute loss
                loss = self.loss(logits, masks.long())

                # backward pass (calculate gradient)
                loss.backward()

                # optimizer
                self.optimizer.step()

                # update statistics
                #    loss
                losses.update(loss.item(), self.batch_size)

                #    metrics
                pbar_odict = OrderedDict(loss=f'{loss.item():1.5f}')
                miou = None
                if self.train_iou:
                    with torch.no_grad():
                        confusion_matrix = confusion_matrix + get_confusion_matrix(preds, masks)
                        miou = get_iou_metrics(confusion_matrix)
                    pbar_odict.update({'mean_iou': f'{miou:1.5f}'})

                pbar.set_postfix(pbar_odict)
                pbar.update(1)

        return losses.avg, miou, pbar.last_print_t - pbar.start_t

    def _validate_epoch(self, loader):

        losses = AverageMeter("val_loss")
        confusion_matrix = np.zeros((2, 2), dtype=np.uint64)

        with tqdm(total=len(loader), desc="Validating", leave=False) as pbar:

            for sample in loader:

                images = sample['image'].cuda(self.device) if self.device.startswith('cuda') else sample['image']
                masks = sample['mask'].cuda(self.device) if self.device.startswith('cuda') else sample['mask']

                # forward pass
                logits = self.net(images)

                # predictions
                if self.net.n_classes == 1:
                    preds = torch.sigmoid(logits)
                else:
                    preds = torch.softmax(logits, dim=1)

                # compute loss
                loss = self.loss(logits, masks.long())

                # update statistics
                #    loss
                losses.update(loss.item(), self.batch_size)

                #    IOU
                confusion_matrix = confusion_matrix + get_confusion_matrix(preds, masks)

                pbar.update(1)
        miou = get_iou_metrics(confusion_matrix)
        return losses.avg, miou
