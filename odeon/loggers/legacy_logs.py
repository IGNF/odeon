"""
Legacy ODEON logger
----------

Logger for export model in old (legacy) ODEON format

"""
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union
from weakref import ReferenceType  # noqa

import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint  # noqa
from pytorch_lightning.loggers.base import (
    DummyExperiment,
    LightningLoggerBase,
    rank_zero_experiment,
)
from pytorch_lightning.utilities import rank_zero_only

log = logging.getLogger(__name__)

INDENT = 2


class OdeonLegacyLogger(LightningLoggerBase):
    r"""
    Logger for export model in old (legacy) ODEON format

    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        log_model: Union[str, bool] = False,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
    ):
        super().__init__()
        self._save_dir = save_dir
        self._log_model = log_model
        self._checkpoint_callbacks = []
        self._logged_model_time = {}
        self._name = name or ""
        self._version = version

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        pass

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        pass

    @property
    @rank_zero_experiment
    def experiment(self) -> DummyExperiment:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment:
            return self._experiment

        self._experiment = DummyExperiment()
        return self._experiment

    @property
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.
        """
        return self._name

    @property
    def version(self) -> int:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment
        """
        return self._version

    @rank_zero_only
    def after_save_checkpoint(
        self, checkpoint_callback: "ReferenceType[ModelCheckpoint]"
    ) -> None:
        # log checkpoints as artifacts
        if (
            self._log_model == "all"
            or self._log_model is True
            and checkpoint_callback.save_top_k == -1
        ):
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            if checkpoint_callback not in self._checkpoint_callbacks:
                self._checkpoint_callbacks.append(checkpoint_callback)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # log checkpoints as artifacts
        for checkpoint_callback in self._checkpoint_callbacks:
            self._scan_and_log_checkpoints(checkpoint_callback)

    @rank_zero_only
    def _scan_and_log_checkpoints(
        self, checkpoint_callback: "ReferenceType[ModelCheckpoint]"
    ) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = {
            checkpoint_callback.last_model_path: checkpoint_callback.current_score,
            checkpoint_callback.best_model_path: checkpoint_callback.best_model_score,
            **checkpoint_callback.best_k_models,
        }
        checkpoints = sorted(
            (Path(p).stat().st_mtime, p, s)
            for p, s in checkpoints.items()
            if Path(p).is_file()
        )
        checkpoints = [
            c
            for c in checkpoints
            if c[1] not in self._logged_model_time.keys()
            or self._logged_model_time[c[1]] < c[0]
        ]

        # log iteratively all new checkpoints
        for t, p, s in checkpoints:
            metadata = {
                "score": s,
                "original_path": Path(p),
                "original_filename": Path(p).name,
                "ModelCheckpoint": {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }

            self._ckpt_to_pth(metadata["original_path"])
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t

    def _ckpt_to_pth(self, ckpt_path: Path, out_path=None):
        if out_path is None:
            out_pth = Path(ckpt_path).with_suffix(".pth")
        else:
            out_pth = out_path

        ckpt_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        state_dict = ckpt_dict["state_dict"]
        # remove criterion/loss info
        for k in state_dict.keys():
            if "criterion" in k:
                state_dict.pop(k, None)
        # remove model prefix in all other keys
        state_dict_keys = list(state_dict.keys())
        for k in state_dict_keys:
            if k.split(".")[0] == "model":
                state_dict[k[6:]] = state_dict.pop(k, None)
        torch.save(state_dict, out_pth)
