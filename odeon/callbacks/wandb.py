import subprocess
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from odeon import LOGGER
from odeon.commons.exception import OdeonError, ErrorCodes
from odeon.commons.metric.plots import plot_confusion_matrix

try:
    import wandb
except ModuleNotFoundError as error:
    LOGGER.error("ERROR: WANDB callbacks have been called but wandb package is not installed")
    raise OdeonError(ErrorCodes.ERR_CALLBACK_ERROR,
                     "something went wrong with WANDB calbacks",
                     stack_trace=error)


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class MetricsWandb(Callback):

    @rank_zero_only
    def add_metrics(self, trainer, pl_module, metric_collection, loss):
        logger = get_wandb_logger(trainer=trainer)
        log_dict = metric_collection.copy()
        log_dict["Loss"] = loss
        del log_dict["cm_macro"]
        del log_dict["cm_micro"]
        logger.log_metrics(log_dict, step=pl_module.current_epoch)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        self.add_metrics(trainer=trainer,
                         pl_module=pl_module,
                         metric_collection=pl_module.val_epoch_metrics,
                         loss=pl_module.val_epoch_loss)


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """
        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder path
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):

                # don't upload files ignored by git
                # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
                command = ["git", "check-ignore", "-q", str(path)]
                not_ignored = subprocess.run(command).returncode == 1

                # don't upload files from .git folder
                not_git = not str(path).startswith(str(git_dir_path))

                if path.is_file() and not_git and not_ignored:
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment
            confusion_matrix = pl_module.val_epoch_metrics["cm_macro"].cpu().numpy()
            fig = plot_confusion_matrix(confusion_matrix,
                                        pl_module.hparams.class_labels)

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"confusion_matrix/{experiment.name}": fig}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()
