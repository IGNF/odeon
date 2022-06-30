import os
from dataclasses import asdict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from odeon.tasks.segmentation_task.data_module import SegmentationTaskDataModule
from odeon.tasks.segmentation_task.model import SemanticSegmentationTaskModel
from odeon.tasks.segmentation_task.callback import LogPredictionsCallback, MetricCallBack
from odeon.core.io_utils import save_dict_as_json


def main():

    print(seg_conf)
    save_dict_as_json(asdict(seg_conf), os.path.join(seg_conf.path_model_output, "hparams.json"))
    data_module = SegmentationTaskDataModule(seg_conf=seg_conf)
    data_module.setup()
    gdf = data_module.db
    test_gdf = data_module.db_test
    model = SemanticSegmentationTaskModel(gdf=gdf,
                                          seg_conf=seg_conf,
                                          test_gdf=test_gdf)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=seg_conf.path_model_log)
    pred_logger = LogPredictionsCallback(seg_conf=seg_conf)
    metric_callback = MetricCallBack(seg_conf=seg_conf)
    # arguments made to CometLogger are passed on to the comet_ml.Experiment class

    """
    comet_logger = pl_loggers.CometLogger(
        api_key="ALpCE5ENfXcq3gmsqs2pfVBiD",
        # save_dir=".",  # Optional
        project_name="poc_gers",  # Optional
        experiment_name="test-comet-2",  # Optional
    )
    """

    # wandb_logger = pl_loggers.WandbLogger(seg_conf.project_name,)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(dirpath=seg_conf.path_model_checkpoint,
                                       save_top_k=seg_conf.save_top_k_models,
                                       filename='epoch-{epoch}-loss-{mean_val_iou:.2f}',
                                       mode="max",
                                       monitor='mean_val_iou')

    callbacks = [lr_monitor, model_checkpoint, pred_logger]
    # callbacks = [model_checkpoint]
    trainer = Trainer(max_epochs=seg_conf.max_epochs,
                      gpus=seg_conf.gpu,
                      callbacks=callbacks,
                      accelerator=None,
                      progress_bar_refresh_rate=2,
                      logger=tb_logger,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0)

    trainer.fit(model=model,
                train_dataloader=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader())

    trainer.test(ckpt_path="best",
                 test_dataloaders=data_module.test_dataloader())


if __name__ == "__main__":

    main()
