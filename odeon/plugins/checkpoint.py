from pytorch_lightning.plugins import CheckpointIO


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        pass

    def load_checkpoint(self, path, storage_options=None):
        pass

    def remove_checkpoint(self, path):
        pass
