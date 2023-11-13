import time
from functools import reduce


class Timer:
    """
    Timer.

    ...

    Attributes
    ----------
    name : string
        name of the portion of code that is monitored.
    start: datetime

    Example
    -------

    with Timer("A"):
        call_to_function_A()
    """

    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ''

    def __enter__(self):
        self.start = time.process_time()

    @staticmethod
    def __seconds_to_str(t):
        return "%02dh:%02dm:%02d.%03ds" % reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(t * 1000,), 1000, 60, 60])

    def __exit__(self, exc_type, exc_value, traceback):
        self.duration = (time.process_time() - self.start)
        print('Code block' + self.name + ' took: ' + self.__seconds_to_str(self.duration))
