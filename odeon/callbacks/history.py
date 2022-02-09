import os
from time import gmtime, strftime
import pytorch_lightning as pl


class HistoryWriter(pl.Callback):

    def __init__(self,
                 path_output=None):
        super().__init__()
        self.path_output = path_output
        print("Init")
        print(self.trainer)

    # def on_batch_end(self, trainer):
