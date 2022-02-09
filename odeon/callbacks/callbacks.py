import os
from time import gmtime, strftime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from odeon.commons.exception import OdeonError, ErrorCodes


def check_path_ckpt(path, description=None): 
    path_ckpt = None
    if not os.path.exists(path):
        path_ckpt = path
    else:
        description = description if description is not None else ""
        path_ckpt = os.path.join(path, description + "_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
        os.makedirs(path_ckpt)
    return path_ckpt

ckpt_descript = f"test_pl"
checkpoint_miou_callback = ModelCheckpoint(monitor="val_miou",
                                          dirpath=check_path_ckpt("odeon_miou_ckpt", description=ckpt_descript),
                                          filename="sample-test-{epoch:02d}-{val_miou:.2f}",
                                          save_top_k=3,
                                          mode="max")

checkpoint_loss_callback = ModelCheckpoint(monitor="val_loss",
                                          dirpath=check_path_ckpt("odeon_loss_ckpt", description=ckpt_descript),
                                          filename="sample-test-{epoch:02d}-{val_loss:.2f}",
                                          save_top_k=3,
                                          mode="min")


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