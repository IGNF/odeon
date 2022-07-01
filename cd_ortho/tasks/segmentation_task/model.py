import os
import pytorch_lightning as pl
from typing import List, Any, Optional, Dict
import torch
import pandas as pd
import seaborn as sn
import torchmetrics
from cd_ortho.models.segmentation_models import SegmentationModelFactory
from geopandas import gpd
from torchmetrics import JaccardIndex, ConfusionMatrix, Accuracy, MetricCollection, Precision, Recall
from cd_ortho.losses.losses import CrossEntropyWithLogitsLoss
from cd_ortho.metrics.metrics_util import AverageMeter
from cd_ortho.core.constants import NOMENCLATURE
from cd_ortho.core.io_utils import save_dict_as_json, create_path_if_not_exists
from cd_ortho.core.rio_utils import get_meta_for_img, save_mask_as_raster


class SemanticSegmentationTaskModel(pl.LightningModule):

    def __init__(self,
                 seg_conf: SegmentationTaskConf,
                 gdf: gpd.GeoDataFrame,
                 test_gdf: gpd.GeoDataFrame
                 ):

        super().__init__()
        self.seg_conf: SegmentationTaskConf = seg_conf
        self.gdf: gpd.GeoDataFrame = gdf
        self.test_gdf: gpd.GeoDataFrame = test_gdf

        model_factory = SegmentationModelFactory(in_chans=len(self.seg_conf.img_bands),
                                                 classes=self.seg_conf.classes,
                                                 model_name=self.seg_conf.model_name,
                                                 encoder_name=self.seg_conf.encoder_name)

        self.model = model_factory.get()
        self.test_output_path = None
        t_max = self.seg_conf.t_max
        eta_min = self.seg_conf.eta_min
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.seg_conf.lr,
                                         momentum=self.seg_conf.momentum,
                                         weight_decay=self.seg_conf.weight_decay)

        """
        if self.seg_conf.pretrained and self.seg_conf.fine_tune:
            pretrained_params = self.model.encoder.parameters()
            new_parameters = self.model
            self.optimizer = torch.optim.Adam([dict(params=self.model.parameters(), lr=self.seg_conf.lr),
                                              dict(params=self.model.parameters(), lr=self.seg_conf.lr)])
        else:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.seg_conf.lr)
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10, verbose=self.seg_conf.debug, cooldown=4)
        self.criterion = CrossEntropyWithLogitsLoss()
        self.train_losses = AverageMeter("avg_train_loss")
        self.val_losses = AverageMeter("avg_val_loss")
        self.classes = [v[1] for k, v in NOMENCLATURE[self.seg_conf.nomenclature].items()]
        self.train_metrics = MetricCollection({"train_accuracy": Accuracy(),
                                               "train_iou": IoU(self.seg_conf.classes, ignore_index=0, absent_score=1.0)})
        self.val_metrics = MetricCollection({"val_accuracy": Accuracy(),
                                             "val_iou": IoU(self.seg_conf.classes, ignore_index=0, absent_score=1.0)})

        self.acc_val_metrics: float = 0.0
        self.acc_train_metrics: float = 0.0
        self.mean_train_metrics = dict()
        self.current_val_preds: Optional[torch.Tensor] = None
        self.val_cm = ConfusionMatrix(self.seg_conf.classes, normalize="true")
        self.test_2019_metrics: Optional[torchmetrics.MetricCollection] = None
        self.test_2016_style_2016_metrics: Optional[torchmetrics.MetricCollection] = None
        self.test_2016_style_2019_metrics: Optional[torchmetrics.MetricCollection] = None
        self.test_change_2019_metrics: Optional[torchmetrics.MetricCollection] = None
        self.test_change_2016_metrics: Optional[torchmetrics.MetricCollection] = None
        self.test_df: Optional[List] = None
        self.test_df_change: Optional[List] = None
        self.path_to_pred_2019: Optional[str] = None
        self.path_to_pred_2016_style_2016: Optional[str] = None
        self.path_to_pred_2016_style_2019: Optional[str] = None
        self.path_to_change_style_2019: Optional[str] = None
        self.path_to_pred_style_2016: Optional[str] = None
        self.metrics_device = torch.device('cuda:0')

    def training_step(self, batch, batch_idx):

        img, target = batch["img"], batch["mask"].squeeze()
        logits = self.model(img)
        pred = torch.softmax(logits, dim=1)
        # backpropagation
        loss = self.criterion(logits, target)
        # LOGGER.info(target.shape)
        # LOGGER.info(target.max())
        # LOGGER.info(target.min())
        self.train_metrics.update(preds=pred, target=target)
        self.train_losses.update(loss, self.seg_conf.train_conf.batch_size)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        imgs, targets, indices = batch["img"], batch["mask"].squeeze(), batch["index"]
        logits: torch.Tensor = self.model(imgs)
        preds: torch.Tensor = torch.softmax(logits, dim=1)
        self.current_val_preds = preds
        self.val_metrics.update(preds=preds, target=targets)
        self.val_cm.update(preds=preds, target=targets)
        val_loss = self.criterion(logits, targets)
        self.val_losses.update(val_loss, self.seg_conf.train_conf.batch_size)
        self.log('val_loss', val_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {"loss": val_loss}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.acc_train_metrics = self.train_metrics.compute()

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.acc_val_metrics = self.val_metrics.compute()
        if self.logger is not None:
            self.logger.experiment.add_scalars("Loss/Avg",
                                               {"avg_train_loss": self.train_losses.get_avg(),
                                                "avg_val_loss": self.val_losses.get_avg()},
                                                global_step=self.current_epoch)
            self.logger.experiment.add_scalars("IOU/Avg",
                                               {f"mean_val_iou": self.acc_val_metrics["val_iou"],
                                                "mean_train_iou": self.acc_train_metrics["train_iou"]},
                                               global_step=self.current_epoch)
            self.logger.experiment.add_scalars("Accuracy/Avg",
                                               {f"mean_val_accuracy": self.acc_val_metrics["val_accuracy"],
                                                "mean_train_accuracy": self.acc_train_metrics["train_accuracy"]},
                                               global_step=self.current_epoch)
            self.log('mean_val_iou', self.acc_val_metrics["val_iou"], on_step=False, on_epoch=True, prog_bar=True, logger=False)
            # Print weights histo
            if (self.current_epoch == 1) or ((self.current_epoch % self.seg_conf.histo_weights_every_n_epochs == 0)
                                             and (self.current_epoch != 0)):
                self.custom_histogram_weights()
        val_cm = self.val_cm.compute().cpu().numpy()
        df_cm = pd.DataFrame(val_cm,
                             index=[i for i in self.classes],
                             columns=[i for i in self.classes])
        plot_cm = sn.heatmap(df_cm, annot=True).get_figure()
        self.logger.experiment.add_figure("Confusion matrix",
                                          plot_cm,
                                          global_step=self.current_epoch)
        self.val_losses.reset()
        self.train_losses.reset()
        self.val_metrics.reset()
        self.train_metrics.reset()
        self.acc_val_metrics = 0.0
        self.acc_train_metrics = 0.0
        self.val_cm.reset()

    def on_test_epoch_start(self) -> None:

        self.test_df = []
        self.test_df_change = []
        self.test_2019_metrics = torchmetrics.MetricCollection({"cm_2019": ConfusionMatrix(self.seg_conf.classes).to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                "iou_2019": IoU(self.seg_conf.classes, ignore_index=0, absent_score=1.0).to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                "accuracy_2019": Accuracy().to(torch.device("cuda", self.seg_conf.gpu[0]))})
        self.test_2016_style_2016_metrics = torchmetrics.MetricCollection({"cm_2016_style_2016": ConfusionMatrix(self.seg_conf.classes).to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                "iou_2016_style_2016": IoU(self.seg_conf.classes, ignore_index=0, absent_score=1.0).to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                "accuracy_2016_style_2016": Accuracy().to(torch.device("cuda", self.seg_conf.gpu[0]))})
        self.test_2016_style_2019_metrics = torchmetrics.MetricCollection({"cm_2016_style_2019": ConfusionMatrix(self.seg_conf.classes).to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                "iou_2016_style_2019": IoU(self.seg_conf.classes, ignore_index=0, absent_score=1.0).to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                "accuracy_2016_style_2019": Accuracy().to(torch.device("cuda", self.seg_conf.gpu[0]))})

        self.test_change_2019_metrics = torchmetrics.MetricCollection({"cm_change_style_2019": ConfusionMatrix(2).to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                       "iou_change_style_2019": IoU(self.seg_conf.classes, ignore_index=0, absent_score=1.0).to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                       "accuracy_change_style_2019": Accuracy().to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                       "precision_change_style_2019": Precision(ignore_index=0,
                                                                                                                mdmc_average='global').to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                       "recall_change_style_2019": Recall(ignore_index=0, mdmc_average='global').to(torch.device("cuda", self.seg_conf.gpu[0]))})
        self.test_change_2016_metrics = torchmetrics.MetricCollection({"cm_change_style_2016": ConfusionMatrix(2).to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                       "iou_change_style_2016": IoU(self.seg_conf.classes, ignore_index=0, absent_score=1.0).to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                       "accuracy_change_style_2016": Accuracy().to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                       "precision_change_style_2016": Precision(ignore_index=0, mdmc_average='global').to(torch.device("cuda", self.seg_conf.gpu[0])),
                                                                       "recall_change_style_2016": Recall(ignore_index=0, mdmc_average='global').to(torch.device("cuda", self.seg_conf.gpu[0]))})

        self.path_to_pred_2019 = os.path.join(self.seg_conf.path_model_tests, "pred_2019")
        self.path_to_pred_2016_style_2016 = os.path.join(self.seg_conf.path_model_tests, "pred_2016_style_2016")
        self.path_to_pred_2016_style_2019 = os.path.join(self.seg_conf.path_model_tests, "pred_2016_style_2019")
        self.path_to_change_style_2019 = os.path.join(self.seg_conf.path_model_tests, "pred_change_2019")
        self.path_to_change_style_2016 = os.path.join(self.seg_conf.path_model_tests, "pred_change_2016")
        create_path_if_not_exists(self.path_to_pred_2019)
        create_path_if_not_exists(self.path_to_pred_2016_style_2019)
        create_path_if_not_exists(self.path_to_pred_2016_style_2016)
        create_path_if_not_exists(self.path_to_change_style_2019)
        create_path_if_not_exists(self.path_to_change_style_2016)

    def test_step(self, batch, batch_idx):

        indexes = batch["index"]
        updateds = batch["updated"]
        imgs_2019 = batch["img"]
        imgs_2016_style_2016 = batch["img_2016_style_2016"]
        imgs_2016_syle_2019 = batch["img_2016_style_2019"]
        masks_2019 = batch["mask"]
        masks_2016 = batch["mask_2016"]
        masks_change = batch["mask_change"]
        preds_2019 = torch.argmax(torch.softmax(self.model(imgs_2019), dim=1), dim=1)
        preds_2016_style_2016 = torch.argmax(torch.softmax(self.model(imgs_2016_style_2016), dim=1), dim=1)
        preds_2016_style_2019 = torch.argmax(torch.softmax(self.model(imgs_2016_syle_2019), dim=1), dim=1)
        preds_change_style_2016 = torch.ne(preds_2019, preds_2016_style_2016).long()
        preds_change_style_2019 = torch.ne(preds_2019, preds_2016_style_2019).long()
        zipper = zip(indexes, updateds, imgs_2019, imgs_2016_style_2016, imgs_2016_syle_2019, masks_change, masks_2019,
                     masks_2016, preds_2019, preds_2016_style_2019, preds_2016_style_2016, preds_change_style_2016,
                     preds_change_style_2019)

        for sample in zipper:

            img_2019, imgs_2016_syle_2016, img_2016_syle_2019 = sample[2], sample[3], sample[4]
            mask_change, mask_2019, mask_2016 = sample[5].squeeze(), sample[6].squeeze(), sample[7].squeeze()
            pred_2019, pred_2016_style_2019, pred_2016_style_2016 = sample[8], sample[9], sample[10]
            # LOGGER.info(pred_2019)
            pred_change_style_2016, pred_change_style_2019 = sample[11], sample[12]
            test_2019_metrics = self.numpyfy_dict(self.test_2019_metrics(preds=pred_2019, target=mask_2019))

            test_2016_style_2016_metrics = self.numpyfy_dict(self.test_2016_style_2016_metrics(preds=pred_2016_style_2016,
                                                                                               target=mask_2016))

            test_2016_style_2019_metrics = self.numpyfy_dict(self.test_2016_style_2019_metrics(preds=pred_2016_style_2019,
                                                                                               target=mask_2016))

            index = int(sample[0].cpu().numpy())
            img_name = self.test_gdf.loc[index, self.seg_conf.test_conf.img_2019_field].split("/")[-1]
            geometry = self.test_gdf.loc[index, "geometry"]
            path_img = os.path.join(self.seg_conf.data_dir, self.test_gdf.loc[index, self.seg_conf.test_conf.img_2019_field])
            meta, profile = get_meta_for_img(path_img)
            profile["count"] = 1
            updated = int(sample[1].cpu().numpy())
            save_mask_as_raster(pred_2019.unsqueeze(dim=0).cpu().numpy(), profile, os.path.join(self.path_to_pred_2019, img_name))
            save_mask_as_raster(pred_2016_style_2016.unsqueeze(dim=0).cpu().numpy(), profile, os.path.join(self.path_to_pred_2016_style_2016, img_name))
            save_mask_as_raster(pred_2016_style_2019.unsqueeze(dim=0).cpu().numpy(), profile, os.path.join(self.path_to_pred_2016_style_2019, img_name))
            save_mask_as_raster(pred_change_style_2016.unsqueeze(dim=0).cpu().numpy(), profile, os.path.join(self.path_to_change_style_2016, img_name))
            save_mask_as_raster(pred_change_style_2019.unsqueeze(dim=0).cpu().numpy(), profile, os.path.join(self.path_to_change_style_2019, img_name))

            d = {"index": index, "updated": updated, "geometry": geometry}
            d.update({k: float(v) for k, v in test_2019_metrics.items() if "cm" not in k})
            d.update({k: float(v) for k, v in test_2016_style_2016_metrics.items() if "cm" not in k})
            d.update({k: float(v) for k, v in test_2016_style_2019_metrics.items() if "cm" not in k})
            self.test_df.append(d)

            if updated == 1:

                test_change_2016_metrics = self.numpyfy_dict(self.test_change_2016_metrics(preds=pred_change_style_2016,
                                                                                           target=mask_change))

                test_change_2019_metrics = self.numpyfy_dict(self.test_change_2019_metrics(preds=pred_change_style_2019,
                                                                                           target=mask_change))

                d = {"index": index, "updated": updated, "geometry": geometry}
                d.update({k: float(v) for k, v in test_change_2016_metrics.items() if "cm" not in k})
                d.update({k: float(v) for k, v in test_change_2019_metrics.items() if "cm" not in k})
                self.test_df_change.append(d)

        """
        for pred_2019, pred_2016_style_2016, pred_2016_style_2019, mask_2019, mask_2016, mask_change in
            zip(preds_2019, preds_2016_style_2016, preds_2016_style_2019, masks_2019, masks_2016, masks_change):
        """



    def test_epoch_end(self, outputs: List[Any]) -> None:

        test_change_2016_metrics: Dict = self.jsonify_dict(self.test_change_2016_metrics.compute())
        test_change_2019_metrics: Dict = self.jsonify_dict(self.test_change_2019_metrics.compute())
        test_2019_metrics: Dict = self.jsonify_dict(self.test_2019_metrics.compute())
        test_2016_style_2019_metrics: Dict = self.jsonify_dict(self.test_2016_style_2019_metrics.compute())
        test_2016_style_2016_metrics: Dict = self.jsonify_dict(self.test_2016_style_2016_metrics.compute())
        d: Dict = dict(test_2019_metrics, **test_2016_style_2016_metrics)
        d.update(test_2016_style_2019_metrics)
        d.update(test_change_2016_metrics)
        d.update(test_change_2019_metrics)
        save_dict_as_json(d, os.path.join(self.seg_conf.path_model_tests, "metrics.json"))
        test_gdf = gpd.GeoDataFrame(self.test_df, crs=self.gdf.crs, geometry="geometry")
        test_change_gdf = gpd.GeoDataFrame(self.test_df_change, crs=self.gdf.crs, geometry="geometry")
        # TODO save as df, save metrics dict and print cm
        test_gdf.to_file(os.path.join(self.seg_conf.path_model_tests, "lcm_metrics_by_patch.geojson"), driver="GeoJSON")
        test_change_gdf.to_file(os.path.join(self.seg_conf.path_model_tests, "change_metrics_by_patch.geojson"), driver="GeoJSON")
        # self.print_cm(test_2019_metrics["cm_2019"], os.path.join(self.seg_conf.path_model_tests, "cm_2019.png"))

    def print_cm(self, cm, output):

        df_cm = pd.DataFrame(cm,
                             index=[i for i in self.classes],
                             columns=[i for i in self.classes])
        plot_cm = sn.heatmap(df_cm, annot=True, cmap='coolwarm', linecolor='white', linewidths=1)
        figure = plot_cm.get_figure()
        figure.savefig(output, dpi=400)

    def configure_optimizers(self):
        """
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                linear_warmup_decay(self.warmup_steps, self.total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }
        """

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
            'monitor': 'avg_val_loss'
        }
        # return [self.optimizer]

    def custom_histogram_weights(self):

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def numpyfy_dict(self, d: Dict[str, torch.Tensor]):

        out = {k: v.cpu().numpy() for k, v in d.items()}
        return out

    def jsonify_dict(self, d: Dict[str, torch.Tensor]):

        out = {k: v.cpu().numpy().tolist() if ("cm" in k) else float(v.cpu().numpy()) for k, v in d.items()}
        return out