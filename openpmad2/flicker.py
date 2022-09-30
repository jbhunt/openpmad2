import numpy as np
from psychopy.visual import GratingStim
from openpmad2.bases import StimulusBase

class FullFieldFlicker(StimulusBase):
    """
    """

    def __init__(self, display):
        """
        """

        super().__init__(display)

        return

    def present(
        self,
        period=2,
        duration=30,
        levels=(-1, 1),
        textureShape=(16,16)
        ):
        """
        """

        flicker = GratingStim(
            self.display,
            size=self.display.size,
            units=self.display.units,
        )

        #
        nCycles = int(np.ceil(duration / period))
        self.display.idle(3)
        for cycleIndex in range(nCycles):
            for level in levels:
                flicker.tex = np.full(textureShape, level)
                self.display.signalEvent(3, units='frames')
                for frameIndex in range(round(period / 2 * self.display.fps)):
                    flicker.draw()
                    self.display.flip()

        #
        self.display.signalEvent(0.05, units='seconds')
        self.display.idle(3, units='seconds')

        return