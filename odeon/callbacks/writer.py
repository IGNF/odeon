import os

import rasterio
from pytorch_lightning.callbacks import BasePredictionWriter
from rasterio.warp import aligned_target

from odeon.commons.image import TypeConverter
from odeon.commons.rasterio import ndarray_to_affine

THRESHOLD = 0.5


class PatchPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        output_type,
        write_interval,
        threshold=THRESHOLD,
        img_size_pixel=None,
        sparse_mode=False,
    ):

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
            self.img_size_pixel = min(
                trainer.datamodule.sample_dims["image"][0],
                trainer.datamodule.sample_dims["image"][1],
            )

        self.gdal_options = {
            "compress": "LZW",
            "tiled": True,
            "blockxsize": self.img_size_pixel,
            "blockysize": self.img_size_pixel,
            "SPARSE_MODE": self.sparse_mode,
        }

        self.meta = trainer.datamodule.meta["test"]
        self.meta["driver"] = "GTiff"
        self.meta["dtype"] = (
            "uint8" if self.output_type in ["uint8", "bit"] else "float32"
        )
        self.meta["count"] = trainer.datamodule.num_classes
        self.meta["width"] = self.img_size_pixel
        self.meta["height"] = self.img_size_pixel
        if self.output_type == "bit":
            self.gdal_options["bit"] = 1
        return super().on_predict_start(trainer, pl_module)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):

        probas, filenames, affines = (
            prediction["proba"],
            prediction["filename"],
            prediction["affine"],
        )

        # Pass prediction and their transformations on CPU
        probas = probas.cpu().numpy()
        affines = affines.cpu().numpy()

        for proba, filename, affine in zip(probas, filenames, affines):
            output_file = os.path.join(self.output_dir, filename)
            self.meta["transform"] = ndarray_to_affine(affine)
            self.meta["transform"], _, _ = aligned_target(
                self.meta["transform"],
                self.meta["width"],
                self.meta["height"],
                trainer.datamodule.resolution["test"],
            )

            with rasterio.open(
                output_file, "w", **self.meta, **self.gdal_options
            ) as src:
                converter = TypeConverter()
                pred = (
                    converter.from_type("float32")
                    .to_type(self.output_type)
                    .convert(proba, threshold=self.threshold)
                )
                src.write(pred)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        if not self.interval.on_batch:
            return

        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
        )
