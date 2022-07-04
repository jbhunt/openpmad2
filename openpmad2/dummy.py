from . import bases
import time

class DummyStim(bases.Stimulus):
    """
    """

    def __init__(self, display, delay=5):
        super().__init__(display)
        self.delay = delay
        return

    def make(self):
        return

    def play(self):
        time.sleep(self.delay)
        return
