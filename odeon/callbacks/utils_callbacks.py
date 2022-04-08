import os
from time import gmtime, strftime 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
import rasterio
from rasterio.warp import aligned_target
from odeon.commons.image import TypeConverter
from odeon.commons.rasterio import ndarray_to_affine

THRESHOLD = 0.5


class LightningCheckpoint(ModelCheckpoint):

    def __init__(
        self,
        monitor,
        dirpath,
        save_top_k,
        filename=None,
        version=None,
        **kwargs,
        ):

        self.save_top_k = save_top_k
        if filename is None:
            filename = "checkpoint-{epoch:02d}-{" + monitor + ":.2f}"
        elif self.save_top_k > 1:
            filename = os.path.splitext(filename)[0] + "-{epoch:02d}-{" + monitor + ":.2f}"
        else:
            filename = os.path.splitext(filename)[0]

        self.version = version
        dirpath = self.check_path_ckpt(dirpath)
        super().__init__(monitor=monitor, dirpath=dirpath, filename=filename, save_top_k=save_top_k, **kwargs)

    def check_path_ckpt(self, path):
        if not os.path.exists(path):
            path_ckpt = path if self.version is None else os.path.join(path, self.version)
        else:
            if self.version is None:
                description = "version_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
            else:
                description = self.version
            path_ckpt = os.path.join(path, description)
        return path_ckpt

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        return super().on_load_checkpoint(trainer, pl_module, callback_state)

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)


class HistorySaver(pl.Callback):
    
    def __init__(self):
        super().__init__()
        self.idx_loggers = None

    def on_fit_start(self, trainer, pl_module):
        if pl_module.logger is not None:
            idx_csv_loggers = [idx for idx, logger in enumerate(pl_module.logger.experiment)\
                if isinstance(logger, pl.loggers.csv_logs.ExperimentWriter)]
            if idx_csv_loggers :
                self.idx_loggers = {'val': idx_csv_loggers[0], 'test': idx_csv_loggers[-1]}
            else:
                self.idx_loggers = {'val': None, 'test': None}

    def on_test_start(self, trainer, pl_module):
        if self.idx_loggers is None:
            self.on_fit_start(trainer, pl_module)
        return super().on_test_start(trainer, pl_module)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        logger_idx = self.idx_loggers['val']
        metric_collection = {key: value.cpu().numpy() for key, value in pl_module.val_epoch_metrics.items()}
        metric_collection['loss'] = pl_module.val_epoch_loss.cpu().numpy()
        metric_collection['learning rate'] = pl_module.hparams.lr  # TODO Add learning rate not from hparams
        pl_module.logger[logger_idx].experiment.log_metrics(metric_collection, pl_module.current_epoch)
        pl_module.logger[logger_idx].experiment.save()

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        logger_idx = self.idx_loggers['test']
        metric_collection = {key: value.cpu().numpy() for key, value in pl_module.test_epoch_metrics.items()}
        metric_collection['loss'] = pl_module.test_epoch_loss.cpu().numpy()
        pl_module.logger[logger_idx].experiment.log_metrics(metric_collection, pl_module.current_epoch)
        pl_module.logger[logger_idx].experiment.save()


class CustomPredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir, output_type, write_interval, threshold=THRESHOLD, img_size_pixel=None, sparse_mode=False):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.output_type = output_type
        self.threshold = threshold
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.meta = None
        self.img_size_pixel = img_size_pixel
        self.sparse_mode = sparse_mode

    def on_predict_start(self, trainer, pl_module):
        if self.img_size_pixel is None:
            self.img_size_pixel = min(trainer.datamodule.sample_dims['image'][0],
                                      trainer.datamodule.sample_dims['image'][1])

        self.gdal_options = {"compress": "LZW",
                             "tiled": True,
                             "blockxsize": self.img_size_pixel,
                             "blockysize": self.img_size_pixel,
                             "SPARSE_MODE": self.sparse_mode}

        self.meta = trainer.datamodule.meta["test"]
        self.meta["driver"] = "GTiff"
        self.meta["dtype"] = "uint8" if self.output_type in ["uint8", "bit"] else "float32"
        self.meta["count"] = trainer.datamodule.num_classes
        self.meta["width"] = self.img_size_pixel
        self.meta["height"] = self.img_size_pixel
        if self.output_type == "bit":
            self.gdal_options["bit"] = 1
        return super().on_predict_start(trainer, pl_module)  

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        probas, filenames, affines = prediction["proba"], prediction["filename"], prediction["affine"]
    
        # Pass prediction and their transformations on CPU
        probas = probas.cpu().numpy()
        affines = affines.cpu().numpy()

        for proba, filename, affine in zip(probas, filenames, affines):
            output_file = os.path.join(self.output_dir, filename)
            self.meta["transform"] = ndarray_to_affine(affine)
            self.meta["transform"], _, _ = aligned_target(self.meta["transform"],
                                                          self.meta["width"],
                                                          self.meta["height"],
                                                          trainer.datamodule.resolution["test"])

            with rasterio.open(output_file, "w", **self.meta, **self.gdal_options) as src:
                converter = TypeConverter()
                pred = converter.from_type("float32").to_type(self.output_type).convert(proba,
                                                                                        threshold=self.threshold)
                src.write(pred)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.interval.on_batch:
            return
        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self.write_on_batch_end(trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx)


# Check size of tensors in forward pass
class CheckBatchGradient(pl.Callback):

    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()

        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)

        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")
