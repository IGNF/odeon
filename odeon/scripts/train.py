"""Training

This module trains a semantic segmentation model from sample files.



Example
-------
    Call this module from the root of the project:

    $ odeon train -c src/json/train.json -v

    This will read the configuration from a json file and train a model.
    Model is stored in output_folder in a .pth file.


Notes
-----


"""

import os
import os.path
import csv
import json
from sklearn.model_selection import train_test_split

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from odeon.commons.core import BaseTool
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.logger.logger import get_new_logger, get_simple_handler
from odeon.commons.guard import files_exist, dirs_exist
from odeon.nn.transforms import Compose, Rotation90, Rotation, Radiometry, ToDoubleTensor
from odeon.nn.datasets import PatchDataset
from odeon.nn.training_engine import TrainingEngine
from odeon.nn.models import model_list, build_model, save_model, load_model, get_train_filenames, resume_train_state
from odeon.nn.losses import BCEWithLogitsLoss, CrossEntropyWithLogitsLoss, FocalLoss2d, ComboLoss

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_training")
ch = get_simple_handler()
STD_OUT_LOGGER.addHandler(ch)
INTERRUPTED = "INTERRUPTED.pth"


class Trainer(BaseTool):
    """Main entry point of training tool

    Implements
    ----------
    BaseTool : object
        the abstract class for implementing a CLI tool
    """

    def __init__(self,
                 verbosity,
                 train_file,
                 model_name,
                 output_folder,
                 val_file=None,
                 percentage_val=0.2,
                 image_bands=None,
                 mask_bands=None,
                 model_filename=None,
                 load_pretrained_enc=False,
                 epochs=300,
                 batch_size=16,
                 patience=20,
                 save_history=False,
                 continue_training=False,
                 loss="ce",
                 class_imbalance=None,
                 optimizer="adam",
                 lr=0.001,
                 data_augmentation=None,
                 device=None,
                 reproducible=False
                 ):
        """[summary]

        Parameters
        ----------
        verbosity : boolean
            verbosity of logger
        train_file : str
            CSV file with image files in this first column and mask files in the second
        model_name : str
            name of model within ('lightunet, 'unet', 'resnetx', 'deeplab')
        output_folder : str
            path to output folder
        val_file : str, optional
            CSV file for validation, by default None
        percentage_val : number, optional
            used if val_file is None, by default 0.2
        image_bands : list of int, optional
            list of band indices, by default None
        mask_bands : list of int, optional
            list of band indices, by default None
        model_filename : str, optional
            name of pth file, if None model name will be used, by default None
        load_pretrained_enc : bool, optional
            WIP: load pretrained weights for encoder, by default False
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
        loss : str, optional
            loss function within ('ce', 'bce', 'wce', 'focal', 'combo'), by default "ce"
        class_imbalance : list of number, optional
            weights for weighted-cross entropy loss, by default None
        optimizer : str, optional
            optimizer name within ('adam', 'SGD'), by default "adam"
        lr : number, optional
            start learning rate, by default 0.001
        data_augmentation : list, optional
            list of data augmentation function within ('rotation', 'rotation90', 'radiometry'), by default None
        device : str, optional
            device if None 'cpu' or 'cuda' if available will be used, by default None
        reproducible : bool, optional
            activate training reproducibility, by default False
        """
        self.verbosity = verbosity
        self.model_name = model_name
        self.output_folder = output_folder
        self.reproducible = reproducible
        self.interrupted = INTERRUPTED
        self.last_name = 'LAST.pth'

        if reproducible is True:
            self.random_seed = 2020
        else:
            self.random_seed = None

        # read csv file with columns: image, mask
        self.train_image_files, self.train_mask_files = self.read_csv_sample_file(train_file)
        if val_file:
            self.val_image_files, self.val_mask_files = self.read_csv_sample_file(val_file)
        else:
            self.train_image_files, self.val_image_files, self.train_mask_files, self.val_mask_files = train_test_split(
                self.train_image_files, self.train_mask_files, test_size=percentage_val, random_state=self.random_seed)

        self.model_filename = model_filename if model_filename is not None else f"{model_name}.pth"
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.save_history = save_history
        self.continue_training = continue_training
        self.loss_name = loss
        self.optimizer_name = optimizer
        self.init_lr = lr
        self.class_imbalance = class_imbalance

        # transformations
        if data_augmentation is None:
            data_augmentation = ['rotation90']
        transformation_dict = {
            "rotation90": Rotation90(),
            "rotation": Rotation(),
            "radiometry": Radiometry()
        }
        transformation_conf = data_augmentation
        transformation_keys = transformation_conf if isinstance(transformation_conf, list) else [transformation_conf]

        self.transformation_functions = list({
            value for key, value in transformation_dict.items() if key in transformation_keys
        })
        self.transformation_functions.append(ToDoubleTensor())

        assert self.batch_size <= len(self.train_image_files), "batch_size must be lower than the length of training \
                                                                dataset"
        train_dataset = PatchDataset(self.train_image_files,
                                     self.train_mask_files,
                                     transform=Compose(self.transformation_functions),
                                     image_bands=image_bands,
                                     mask_bands=mask_bands)
        self.train_dataloader = DataLoader(train_dataset,
                                           self.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           drop_last=True)
        val_dataset = PatchDataset(self.val_image_files,
                                   self.val_mask_files,
                                   transform=Compose(self.transformation_functions),
                                   image_bands=image_bands,
                                   mask_bands=mask_bands)
        self.val_dataloader = DataLoader(val_dataset,
                                         self.batch_size,
                                         shuffle=True,
                                         num_workers=8)

        if image_bands is not None:
            self.n_channels = len(image_bands)
        else:
            self.n_channels = self.get_sample_shape(train_dataset)['image'][0]
        if mask_bands is not None:
            self.n_classes = len(mask_bands)
        else:
            self.n_classes = self.get_sample_shape(train_dataset)['mask'][0]

        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        STD_OUT_LOGGER.info(f"""training :
device: {self.device}
model: {self.model_name}
model file: {self.model_filename}
number of classes: {self.n_classes}
number of samples: {len(val_dataset) + len(train_dataset)} (train: {len(train_dataset)}, \
val: {len(val_dataset)})""")

        self.check()
        self.configure()

    def check(self):

        if self.model_name not in model_list:

            raise OdeonError(message=f"the model name {self.model_name} does not exist",
                             error_code=ErrorCodes.ERR_MODEL_ERROR)

        try:
            files_exist(self.train_image_files)
            files_exist(self.train_mask_files)
            files_exist(self.val_image_files)
            files_exist(self.val_mask_files)
            dirs_exist([self.output_folder])
            if self.continue_training:
                train_files, continue_train = self.get_train_filenames()

                STD_OUT_LOGGER.info(f"initialize model with {train_files['model']}")
                if continue_train:
                    check_train_files = [train_files["model"], train_files["optimizer"]]
                    if self.save_history:
                        check_train_files.append(train_files["history"])
                    files_exist(check_train_files)
                else:
                    STD_OUT_LOGGER.info("continue training not existing models founds, get back to initializing model")

        except OdeonError as error:

            raise OdeonError(ErrorCodes.ERR_TRAINING_ERROR,
                             "something went wrong during training configuration",
                             stack_trace=error)

    def configure(self):

        train_files, continue_train = self.get_train_filenames()
        if not continue_train:
            self.model = build_model(self.model_name, self.n_channels, self.n_classes)
        else:
            self.model = load_model(self.model_name, train_files["model"], self.n_channels, self.n_classes)

        self.optimizer_function = self.get_optimizer(self.optimizer_name, self.model, self.init_lr)
        lr_scheduler = ReduceLROnPlateau(
            self.optimizer_function,
            'min',
            factor=0.5,
            patience=10,
            verbose=self.verbosity,
            cooldown=4,
            min_lr=1e-7)

        if continue_train:
            resume_train_state(
                self.output_folder, self.model_filename, optimizer=self.optimizer_function, scheduler=lr_scheduler)

        loss_function = self.get_loss(self.loss_name, class_weight=self.class_imbalance)

        self.trainer = TrainingEngine(self.model,
                                      loss_function,
                                      self.optimizer_function,
                                      lr_scheduler,
                                      self.output_folder,
                                      self.model_filename,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      patience=self.patience,
                                      save_history=self.save_history,
                                      continue_training=continue_train,
                                      reproducible=self.reproducible,
                                      device=self.device,
                                      verbose=self.verbosity)

        net_params = sum(p.numel() for p in self.model.parameters())

        if continue_train:
            with open(train_files["history"], 'r') as file:
                self.trainer.history.history_dict = json.load(file)

        STD_OUT_LOGGER.info(f"Model parameters trainable : {net_params}")

    def __call__(self):
        """Call the Trainer
        """

        try:
            self.trainer.run(self.train_dataloader, self.val_dataloader)
            # if continue training and stopped cause by patience save 'LAST' model
            if self.continue_training:
                model_filepath = save_model(
                    self.output_folder, f'{self.last_name}', self.trainer.net, optimizer=self.trainer.optimizer,
                    scheduler=self.trainer.lr_scheduler)
                STD_OUT_LOGGER.info(f"Save '{self.last_name}' model : {model_filepath}")
                last_filenames = get_train_filenames(self.output_folder, f'{self.last_name}')
                if self.trainer.save_history:
                    self.trainer.history.save(last_filenames["history"])

        except OdeonError as error:

            raise error

        except KeyboardInterrupt:
            tmp_file = save_model(
                self.output_folder, f'{self.interrupted}', model=self.model, optimizer=self.optimizer_function,
                scheduler=self.trainer.lr_scheduler)
            STD_OUT_LOGGER.info(f"Saved interrupt as {tmp_file}")
            # save also history file to reload epoch
            if self.save_history:
                train_files = get_train_filenames(self.output_folder, f'{self.interrupted}')
                history = self.trainer.history
                history.save(out_file=train_files["history"])

    def read_csv_sample_file(self, file_path):
        """Read a sample CSV file and return a list of image files and a list of mask files.
        CSV file should contain image pathes in the first column and mask pathes in the second.

        Parameters
        ----------
        file_path : str
            path to sample CSV file

        Returns
        -------
        Tuple[list, list]
            a list of image pathes and a list of mask pathes
        """
        image_files = []
        mask_files = []
        if not os.path.exists(file_path):

            raise OdeonError(ErrorCodes.ERR_FILE_NOT_EXIST,
                             f"file ${file_path} does not exist.")

        with open(file_path) as csvfile:
            sample_reader = csv.reader(csvfile)
            for item in sample_reader:
                image_files.append(item[0])
                mask_files.append(item[1])
        return image_files, mask_files

    def get_optimizer(self, optimizer_name, model, lr):
        """Initialize optimizer object from name

        Parameters
        ----------
        optimizer_name : str
            optimizer name possible values = ("adam", "SGD")
        model : nn.Module
            pytorch neural network object
        lr : float
            learning rate
        Returns
        -------
            torch.Optimizer
        """
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            return optim.SGD(model.parameters(), lr=lr)

    def get_loss(self, loss_name, class_weight=None, use_cuda=False):
        """Initialize loss class instance
        Loss function applied directly on models raw prediction (logits)

        Parameters
        ----------
        loss_name : str
            loss name, possible values = ("ce", "bce", "focal", "combo")
        class_weight : list of float, optional
            weights applied to each class in loss computation, by default None
        use_cuda : bool, optional
            use CUDA, by default False

        Returns
        -------
        [type]
            [description]
        """

        if loss_name == "ce":
            if class_weight is not None:
                STD_OUT_LOGGER.info(f"Weights used: {class_weight}")
                weight = torch.FloatTensor(class_weight)
                if use_cuda:
                    weight = weight.cuda()
                return CrossEntropyWithLogitsLoss(weight=weight)

            else:

                return CrossEntropyWithLogitsLoss()

        elif loss_name == "bce":
            return BCEWithLogitsLoss()
        elif loss_name == "focal":
            return FocalLoss2d()
        elif loss_name == "combo":
            return ComboLoss({'bce': 0.75, 'jaccard': 0.25})

    def get_sample_shape(self, dataset):
        """get sample shape from dataloader

        Parameters
        ----------

        dataset : :class:`Dataset`

        Returns
        -------
        tuple
            width, height, n_bands
        """

        sample = dataset.__getitem__(0)

        return {'image': sample['image'].shape, 'mask': sample['mask'].shape}

    def get_train_filenames(self):

        train_files = get_train_filenames(self.output_folder, self.model_filename)
        if self.continue_training:
            if os.path.exists(train_files["model"]):
                model_modif_date = os.path.getmtime(train_files["model"])
            else:
                model_modif_date = os.path.getmtime(self.output_folder)

            interrupted_files = get_train_filenames(self.output_folder, self.interrupted)
            if os.path.exists(interrupted_files["model"]):
                interrupted_modif_date = os.path.getmtime(interrupted_files["model"])
                if interrupted_modif_date > model_modif_date:
                    train_files = interrupted_files
                    model_modif_date = interrupted_modif_date

            last_files = get_train_filenames(self.output_folder, self.last_name)
            if os.path.exists(last_files["model"]):
                last_modif_date = os.path.getmtime(last_files["model"])
                if last_modif_date > model_modif_date:
                    train_files = last_files
                    model_modif_date = last_modif_date

        continue_train = False
        if self.continue_training and os.path.exists(train_files["model"]):
            continue_train = True

        return train_files, continue_train
