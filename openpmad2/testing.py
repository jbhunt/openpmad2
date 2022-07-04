import time
import numpy as np
import multiprocessing as mp

class DummyDeepLabCutProcess(mp.Process):
    """
    """

    def __init__(self, flag, duration=0.1, trange=(0.5, 10), silent=False):
        """
        """

        super().__init__()
        self.flag = flag
        self.started = mp.Value('i', 1)
        self.trange = trange
        self.duration = duration
        self.silent = silent

        return


    def run(self):
        """
        """

        delay = np.random.uniform(self.trange[0], self.trange[1], size=1).item()

        while self.started.value:
            if self.silent:
                continue
            t0 = time.time()
            while time.time() - t0 < delay - self.duration:
                continue
            self.flag.value = 1
            t0 = time.time()
            while time.time() - t0 < self.duration:
                continue
            self.flag.value = 0
            delay = np.random.uniform(self.trange[0], self.trange[1], size=1).item()

        return

    def join(self):
        """
        """

        self.started.value = 0
        super().join()
        return
