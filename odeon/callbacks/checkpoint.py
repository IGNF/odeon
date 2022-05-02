import os
from time import gmtime, strftime 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

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
