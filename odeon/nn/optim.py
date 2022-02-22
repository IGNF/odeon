from pathlib import Path
import torch
from torch import optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ReduceLROnPlateau
)

PATIENCE = 30
FACTOR = 0.5
COOLDOWN = 4
MIN_LR = 1e-7


def build_optimizer(params, learning_rate, optimizer_config=None, resume_files=None):
    optimizer = None
    if optimizer_config is None:
        optimizer = optim.SGD(params, lr=learning_rate)
    else:
        if optimizer_config["optimizer"].lower() == 'adam':
            optimizer= optim.Adam(params, 
                                  lr=learning_rate,
                                  betas=tuple(optimizer_config["betas"]),
                                  eps=optimizer_config["eps"],
                                  weight_decay=optimizer_config["weight_decay"],
                                  amsgrad=optimizer_config["amsgrad"])

        elif optimizer_config["optimizer"].lower() == 'radam':
            optimizer= optim.RAdam(params, 
                                   lr=learning_rate,
                                   betas=tuple(optimizer_config["betas"]),
                                   eps=optimizer_config["eps"],
                                   weight_decay=optimizer_config["weight_decay"])

        elif optimizer_config["optimizer"].lower() == 'adamax':
            optimizer= optim.Adamax(params, 
                                    lr=learning_rate,
                                    betas=tuple(optimizer_config["betas"]),
                                    eps=optimizer_config["eps"],
                                    weight_decay=optimizer_config["weight_decay"])

        elif optimizer_config["optimizer"].lower() == 'sgd':
            optimizer = optim.SGD(params,
                                  lr=learning_rate,
                                  momentum=optimizer_config["momentum"],
                                  dampening=optimizer_config["dampening"],
                                  weight_decay=optimizer_config["weight_decay"],
                                  nesterov=optimizer_config["nesterov"])

        elif optimizer_config["optimizer"].lower() == 'rmsprop':
            optimizer = optim.RMSprop(params,
                                      lr=learning_rate,
                                      alpha=optimizer_config["alpha"],
                                      eps=optimizer_config["eps"],
                                      weight_decay=optimizer_config["weight_decay"],
                                      momentum=optimizer_config["momentum"],
                                      centered=optimizer_config["centered"])

    if resume_files is not None:
        optimizer.load_state_dict(torch.load(resume_files["optimizer"]))

    return optimizer


def build_scheduler(optimizer, scheduler_config=None, patience=None, resume_files=None):
    scheduler = None
    if scheduler_config is None:
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      mode='min',
                                      factor=FACTOR,
                                      patience=patience if patience is not None else PATIENCE,
                                      cooldown=COOLDOWN,
                                      min_lr=MIN_LR)
    else:
        if scheduler_config["scheduler"].lower() == 'reducelronplateau':
            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          mode=scheduler_config["mode"],
                                          factor=scheduler_config["factor"],
                                          patience=scheduler_config["patience"],
                                          cooldown=scheduler_config["cooldown"],
                                          min_lr=scheduler_config["min_lr"])

        elif scheduler_config["scheduler"].lower() == 'cycliclr':
            scheduler = CyclicLR(optimizer=optimizer,
                                 base_lr=scheduler_config["base_lr"],
                                 max_lr=scheduler_config["max_lr"],
                                 step_size_up=scheduler_config["step_size_up"],
                                 step_size_down=scheduler_config["step_size_down"])

        elif scheduler_config["scheduler"].lower() == 'cosineannealinglr':
            scheduler = CosineAnnealingLR(optimizer=optimizer,
                                          T_max=scheduler_config["T_max"],
                                          eta_min=scheduler_config["eta_min"],
                                          last_epoch=scheduler_config["last_epoch"])

        elif scheduler_config["scheduler"].lower() == 'cosineannealingwarmrestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                    T_0=scheduler_config["T_0"],
                                                    T_mult=scheduler_config["T_mult"],
                                                    eta_min=scheduler_config["eta_min"],
                                                    last_epoch=scheduler_config["last_epoch"])

    if resume_files is not None and Path(resume_files["train"]).exists():
        resume_dict = torch.load(resume_files["train"])
        scheduler.load_state_dict(resume_dict["scheduler"])

    return scheduler
