import os
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import pandas as pd
import seaborn as sn
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from odeon.core.image import tensor_to_image
from odeon.metrics.metrics_util import plot_image_debug_validation_loop
from odeon.core.io_utils import create_folder
from odeon import LOGGER
from kornia.augmentation import Denormalize
import matplotlib.patches as mpatches


class LogPredictionsCallback(Callback):

    def __init__(self,
                 mean: Optional[List],
                 std: Optional[List],
                 output_path: Union[str, Path],
                 nomenclature: Dict,
                 colormap,
                 print_n_batches_every_n_epochs: Tuple[int, int]

                 ):

        super(LogPredictionsCallback, self).__init__()
        self._to_denorm = False
        if mean is not None and std is not  None:
            self._to_denorm = True
            self._denorm = Denormalize(mean=torch.from_numpy(np.asarray(mean)),
                                      std=torch.from_numpy(np.asarray(std)))
        self.output_path = output_path
        self.color_map = colormap
        self.nomenclature: nomenclature
        self.patches = []
        self.color_patches: List[mpatches.Patch] = [mpatches.Patch(color=(v[0][0] / 255, v[0][1] / 255, v[0][2] / 255),
                                                                   label=f'{k}-{v[1]}') for k, v in nomenclature.items()]
        self.print_n_batches_every_n_epochs = print_n_batches_every_n_epochs

    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: Any,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int) -> None:
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        # Let's log 20 sample image predictions from first batch
        if batch_idx <= self.print_n_batches_every_n_epochs[0]:

            if (int(pl_module.current_epoch) % self.print_n_batches_every_n_epochs[1] == 0) or \
                    pl_module.current_epoch < 10:
                assert pl_module.current_val_preds is not None

                preds = pl_module.current_val_preds
                imgs, targets, indices = batch["img"], batch["mask"].squeeze(), batch["index"]
                # LOGGER.info(f"imgs shape {imgs.shape}")
                preds = torch.argmax(preds, dim=1)

                for index, img, target, pred in zip(indices, imgs, targets, preds):
                    # LOGGER.info(f"img shape {img.shape}")
                    # LOGGER.info(f"target shape {target.shape}")
                    # LOGGER.info(f"pred shape {pred.shape}")
                    # LOGGER.info(img.shape)
                    img = self._denorm(img) if self._to_denorm else img
                    img = tensor_to_image(img.squeeze())[:, :, 0:3]
                    img = (img * 255.0).astype("uint8")
                    target = target.cpu().numpy()
                    pred = pred.cpu().numpy()
                    index = index.cpu().numpy()
                    name = f"epoch{pl_module.current_epoch}_img_idx_{str(index)}.png"
                    output_dir = os.path.join(self.output_path, f"img_idx_{str(index)}")
                    if os.path.isdir(output_dir) is False:
                        create_folder(output_dir, parents=False, exist_ok=False)
                    output_file = os.path.join(output_dir, name)
                    plot_image_debug_validation_loop(img, target, pred, self.color_map, output_file, patches=self.color_patches)


class MetricCallBack(Callback):

    def __init__(self,
                 nomenclature: Dict):
        super(MetricCallBack, self).__init__()
        self.classes = [v[1] for k, v in nomenclature.items()]

    def on_validation_epoch_end(self, trainer, pl_module: LightningModule) -> None:

        val_cm = pl_module.val_cm.compute().cpu().numpy()
        LOGGER.info(val_cm)
        df_cm = pd.DataFrame(val_cm, index=[i for i in self.classes],
                             columns=[i for i in self.classes])
        plot_cm = sn.heatmap(df_cm, annot=True).get_figure()
        pl_module.logger.experiment.add_figure("Confusion matrix", plot_cm, global_step=pl_module.current_epoch)
        val_cm.reset()
