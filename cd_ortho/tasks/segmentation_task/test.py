import os
from dataclasses import asdict
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from cd_ortho.core.data_module import SegmentationTaskDataModule
from cd_ortho.tasks.segmentation_task.model import SemanticSegmentationTaskModel
from cd_ortho.tasks.segmentation_task.callback import LogPredictionsCallback, MetricCallBack
from cd_ortho.core.io_utils import save_dict_as_json


def main():

    checkpoint = "/var/data/dl/gers/log/segmentation/segmentation_mono_temporelle_multi_style_training_2016_balanced-unet-efficient_b4-soft_aug-sgd-RGB/checkpoint/epoch-epoch=374-loss-mean_val_iou=0.61.ckpt"
    seg_conf = SegmentationTaskConf()
    print(seg_conf)
    save_dict_as_json(asdict(seg_conf), os.path.join(seg_conf.path_model_output, "hparams.json"))
    data_module = SegmentationTaskDataModule(seg_conf=seg_conf)
    data_module.setup(stage="test")
    gdf = data_module.db_test
    test_gdf = data_module.db_test
    model = SemanticSegmentationTaskModel.load_from_checkpoint(gdf=gdf, seg_conf=seg_conf, checkpoint_path=checkpoint,
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
    # callbacks = [model_checkpoint]

    trainer = Trainer(
                      gpus=seg_conf.gpu,
                      accelerator=None,
                      progress_bar_refresh_rate=2)

    trainer.test(test_dataloaders=data_module.test_dataloader(),
                 model=model)


if __name__ == "__main__":

    main()
