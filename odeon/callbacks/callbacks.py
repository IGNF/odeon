import os
from time import gmtime, strftime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Callbacks
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

# Add histograms to tensorboards
class InputMonitor(pl.Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram("target", y, global_step=trainer.global_step)

            
# Launch eval at a precise number of step
class ValEveryNSteps(pl.Callback):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            trainer.run_evaluation(test_mode=False)


# Make a checkpoint at a precise number of step
class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


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