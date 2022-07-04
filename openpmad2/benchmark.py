from . import bases
from . import writing
import numpy as np
from psychopy.visual.movie3 import MovieStim3

class RefreshRateBenchmark(bases.Stimulus):
    """
    """

    def __init__(
        self,
        display,
        ncalls=2,
        duration=3,
        ):
        """
        """

        super().__init__(display)

        self._ncalls = ncalls
        self._duration = duration

        return

    def construct(self, filename=None, metadata=None, prestimulus=1):
        """
        """

        #
        if filename is not None:
            writer = writing.SKVideoVideoWriterWrapper(self.display, filename)
            self._movie = filename

        # Short pre-stimulus epoch
        self.display.background.state = False
        self.display.background.draw()
        warped = self.display.getMovieFrame()
        for iframe in range(int(np.ceil(prestimulus * self.display.fps))):
            if online:
                self.display.flip()
            if filename is not None:
                writer.write(warped)

        for iframe in range(int(np.ceil(self.display.fps * duration))):
            self.display.background.state = not self.display.background.state
            self.display.background.draw()
            warped = self.display.getMovieFrame()
            if online:
                self.display.flip()
            if filename is not None:
                writer.write(warped)

        # Short post-stimulus epoch
        self.display.background.state = False
        self.display.background.draw()
        warped = self.display.getMovieFrame()
        for iframe in range(int(np.ceil(prestimulus * self.display.fps))):
            if online:
                self.display.flip()
            if filename is not None:
                writer.write(warped)

        #
        if filename is not None:
            writer.close()

        return

    def present(self, filename):
        """
        """

        stim = MovieStim3(self.display, filename, units=self.display.units, size=self.display.size)
        stim.play()
        nframes = int(np.ceil(stim.duration / stim._frameInterval))
        for iframe in range(nframes):
            for icall in range(self._ncalls):
                stim.draw()
            self.window.flip()

        return
