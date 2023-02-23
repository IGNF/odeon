from .singleton import Singleton


class App(metaclass=Singleton):
    """ abstract base class for any Odeon App like fit, feature, etc.
    Odeon apps are Singleton"""
    def run(self, *args, **kwargs):
        ...
