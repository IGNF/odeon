

class TrainingEngine:
    def __init__(self, model, epochs=300, **kwargs):
        """TrainingEngine class

        Parameters
        ----------
        model : nn.Module
            model instance derived from a pytorch nn.Module
        epochs : integer
            number of epochs
        """
        self.model = model
        self.epochs = epochs

    def train(self, train_dataloader, val_dataloader):

        pass
