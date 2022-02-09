import os
from time import gmtime, strftime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from odeon.commons.exception import OdeonError, ErrorCodes


class MyModelCheckpoint(ModelCheckpoint):

    def __init__(self, monitor, dirpath, filename=None, **kwargs):
        if filename is None:
            filename = "checkpoint-{epoch:02d}-{" + monitor + ":.2f}"
        dirpath = self.check_path_ckpt(dirpath)
        super().__init__(monitor=monitor, dirpath=dirpath, filename=filename, **kwargs)

    @staticmethod
    def check_path_ckpt(path): 
        if not os.path.exists(path):
            path_ckpt = path
        else:
            description = "version_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
            path_ckpt = os.path.join(path, description)
        return path_ckpt


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