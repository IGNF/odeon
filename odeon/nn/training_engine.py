from collections import OrderedDict
from tqdm import tqdm

import torch

from odeon.nn.history import History
from odeon.commons.metrics import AverageMeter, get_confusion_matrix_torch, get_iou_metrics_torch
from odeon.nn.models import save_model, get_train_filenames

from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes


class TrainingEngine:
    """Training class

    **Continue training :**
    Model and training metadata are saved on files with *LAST* prefix if training is
    stopped because of early_stopping (patience) or number of epochs conditions.
    In case of unwanted training stopping (Ctrl-C /keyboard interrupted) models and
    training information are saved into  files with *INTERUPTED* prefix.
    Training is recover from the last modified models files.

    Parameters
    ----------
    model : :class:`nn.Module`
        pytorch model
    loss : :class:`nn.Module`
        loss class
    optimizer : :class:`Optimizer`
        optimizer
    lr_scheduler : object
        learning rate scheduler
    output_folder : str
        path to output folder
    output_filename : str
        output file name for pth file
    epochs : int, optional
        number of epochs, by default 300
    batch_size : int, optional
        batch size, by default 16
    patience : int, optional
        maximum number of epoch without improvement before train is stopped, by default 20
    save_history : bool, optional
        activate history storing, by default False
    continue_training : bool, optional
        resume a training, by default False
    device : str, optional
        device if None 'cpu' or 'cuda' if available will be used, by default None
    reproducible : bool, optional
        activate training reproducibility, by default False
    verbose : bool, optional
        verbosity, by default False

    Raises
    ------
    OdeonError
        ERR_TRAINER_ERROR,
    """

    def __init__(self, model, loss, optimizer, lr_scheduler, output_folder, output_filename,
                 epochs=300, batch_size=16, patience=20, save_history=False, continue_training=False,
                 device=None, reproducible=False, verbose=False):

        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = model.cuda(self.device) if self.device.startswith('cuda') else model
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.save_history = save_history
        self.continue_training = continue_training
        self.optimizer = optimizer
        self.loss = loss.cuda(self.device) if self.device.startswith('cuda') else loss
        self.lr_scheduler = lr_scheduler
        self.output_folder = output_folder
        self.output_filename = output_filename
        self.train_iou = verbose
        self.multilabel = False
        self.micro_iou = True

        # history
        train_files_dict = get_train_filenames(self.output_folder, self.output_filename)
        try:

            self.history = History(train_files_dict["base"], update=self.continue_training, train_iou=self.train_iou)

        except OdeonError as error:
            raise OdeonError(ErrorCodes.ERR_TRAINER_ERROR,
                             "something went wrong during training",
                             call_stack=error)

    def run(self, train_loader, val_loader):

        LOGGER.info(f'''Training:
            Model: {type(self.net).__name__}
            Batch size: {self.batch_size}
            Loss function: {type(self.loss).__name__}
            Optimizer: {type(self.optimizer).__name__}
            Learning rate: {type(self.lr_scheduler).__name__} starting at {self.optimizer.param_groups[0]['lr']}
            Save history: {self.save_history}
        ''')

        epoch_start = 0
        prec_val_loss = 1000
        patience_counter = 0
        if self.continue_training:
            epoch_start = self.history.get_current_epoch(default=0) + 1
            # in case of interrupted model the last loss in not the last min i.e prec_val_loss
            # and we could recompute patience_counter by number of epoch since the min loss value
            all_val_loss = self.history.get_val_losses()
            if all_val_loss:
                prec_val_loss = min(all_val_loss)
                patience_counter = len(all_val_loss) - all_val_loss.index(prec_val_loss) - 1

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
            self.history.update(epoch, avg_time, train_loss, val_loss,
                                self.optimizer.param_groups[0]['lr'], val_miou, train_mean_iou=train_miou)

            # save model if val_loss has decreased
            if prec_val_loss > val_loss:
                model_filepath = save_model(
                    self.output_folder, self.output_filename, self.net, optimizer=self.optimizer,
                    scheduler=self.lr_scheduler)
                LOGGER.info(f"Saving {model_filepath}")

                if self.save_history:
                    self.history.save()
                    self.history.plot()

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
        use_cuda = True if self.device.startswith('cuda') else False
        if self.multilabel:
            confusion_matrix = torch.zeros((self.net.n_classes, 2, 2), dtype=torch.long)
        else:
            confusion_matrix = torch.zeros((self.net.n_classes, self.net.n_classes), dtype=torch.long)
        if use_cuda:
            confusion_matrix = confusion_matrix.cuda()

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
                loss = self.loss(logits, masks)
                # loss = self.loss(logits, masks.long())

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
                        confusion_matrix = confusion_matrix + get_confusion_matrix_torch(
                            preds, masks, multilabel=self.multilabel, cuda=use_cuda)
                        miou = get_iou_metrics_torch(confusion_matrix, micro=self.micro_iou, cuda=use_cuda)
                    pbar_odict.update({'mean_iou': f'{miou:1.5f}'})

                pbar.set_postfix(pbar_odict)
                pbar.update(1)

        return losses.avg, miou, pbar.last_print_t - pbar.start_t

    def _validate_epoch(self, loader):

        losses = AverageMeter("val_loss")
        # confusion_matrix_np = np.zeros((2, 2), dtype=np.uint64)
        use_cuda = True if self.device.startswith('cuda') else False
        if self.multilabel:
            confusion_matrix = torch.zeros((self.net.n_classes, 2, 2), dtype=torch.long)
        else:
            confusion_matrix = torch.zeros((self.net.n_classes, self.net.n_classes), dtype=torch.long)
        if use_cuda:
            confusion_matrix = confusion_matrix.cuda()

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
                loss = self.loss(logits, masks)

                # update statistics
                #    loss
                losses.update(loss.item(), self.batch_size)

                #    IOU
                confusion_matrix = confusion_matrix + get_confusion_matrix_torch(
                    preds, masks, multilabel=self.multilabel, cuda=use_cuda)

                pbar.update(1)

        # miou_np = get_iou_metrics(confusion_matrix_np)
        miou = get_iou_metrics_torch(confusion_matrix, micro=self.micro_iou, cuda=use_cuda)
        return losses.avg, miou
