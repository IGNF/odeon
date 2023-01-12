from abc import ABC


class App(ABC):
    """ abstract base class for any Odeon App like fit, feature, etc."""
    def run(self):
        ...
