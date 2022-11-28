"""
Experiment name: organization/project/team/phase/experiment_name/run_id
"""

from typing import Protocol

import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger


class MlFlowProtocol(Protocol):

    def configure_mlflow(self, uri, experiment_name: str, run_name: str) -> MLFlowLogger:
        mlflow.set_tracking_uri(uri=uri)
        mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.pytorch.autolog()
        mlflow.start_run(run_name=run_name)
        mlf_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=mlflow.active_run().info.run_id,
        )
        return mlf_logger


class OdnTrainer(Trainer):

    def __init__(self):
        super(OdnTrainer, self).__init__()
