import os

import rasterio
from pytorch_lightning.callbacks import BasePredictionWriter
from rasterio.features import geometry_window
from rasterio.plot import reshape_as_raster
from rasterio.windows import transform

from odeon.commons.image import TypeConverter, substract_margin
from odeon.commons.shape import create_polygon_from_bounds

THRESHOLD = 0.5


class ZonePredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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
        probas, indices = (prediction["proba"], prediction["index"])
        # Pass prediction and their indices on CPU
        probas = probas.cpu().numpy()
        indices = indices.cpu().numpy()
        self.write_preds_by_indices(
            trainer=trainer, predictions=probas, indices=indices
        )

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not self.interval.on_batch:
            return
        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
        )

    def write_preds_by_indices(self, trainer, predictions, indices):
        dm = trainer.datamodule
        for prediction, index in zip(predictions, indices):
            prediction = prediction.transpose((1, 2, 0)).copy()
            prediction = substract_margin(
                img=prediction, margin_x=dm.margin, margin_y=dm.margin
            )
            prediction = reshape_as_raster(prediction)
            converter = TypeConverter()
            prediction = (
                converter.from_type("float32")
                .to_type(dm.output_type)
                .convert(prediction, threshold=dm.threshold)
            )
            output_id = dm.job.get_cell_at(index[0], "output_id")
            name = str(output_id) + ".tif"
            output_file = os.path.join(self.output_dir, name)

            if (
                dm.out_dalle_size is not None
                and dm.rio_ds_collection.collection_has_key(output_id)
            ):
                out = dm.rio_ds_collection.get_rio_dataset(output_id)
            else:
                left = dm.job.get_cell_at(index[0], "left_o")
                bottom = dm.job.get_cell_at(index[0], "bottom_o")
                right = dm.job.get_cell_at(index[0], "right_o")
                top = dm.job.get_cell_at(index[0], "top_o")

                geometry = create_polygon_from_bounds(left, right, bottom, top)
                window = geometry_window(
                    dm.dst, [geometry], pixel_precision=6
                ).round_shape(op="ceil", pixel_precision=4)

                dm.meta_output["transform"] = transform(window, dm.dst.transform)
                out = rasterio.open(
                    output_file, "w+", **dm.meta_output, **dm.gdal_options
                )
                dm.rio_ds_collection.add_rio_dataset(output_id, out)

            left = dm.job.get_cell_at(index[0], "left")
            bottom = dm.job.get_cell_at(index[0], "bottom")
            right = dm.job.get_cell_at(index[0], "right")
            top = dm.job.get_cell_at(index[0], "top")
            geometry = create_polygon_from_bounds(left, right, bottom, top)
            window = geometry_window(out, [geometry], pixel_precision=6).round_shape(
                op="ceil", pixel_precision=4
            )
            indices = [i for i in range(1, dm.num_classes + 1)]
            out.write_band(
                [i for i in range(1, dm.num_classes + 1)], prediction, window=window
            )
            dm.job.set_cell_at(index[0], "job_done", 1)
            if dm.out_dalle_size is not None and dm.job.job_finished_for_output_id(
                output_id
            ):
                dm.rio_ds_collection.delete_key(output_id)
                dm.job.mark_dalle_job_as_done(output_id)
                dm.job.save_job()
            if dm.out_dalle_size is None:
                out.close()
