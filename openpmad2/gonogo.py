import numpy as np
import pathlib as pl
from psychopy import visual
from psychopy import core

class StaticGratingWithProbe():
    """
    """

    def __init__(self, display):
        self.display = display
        self.metadata = None
        return

    def present(
        self,
        probeDuration=1,
        spatialFrequency=0.08,
        probeContrast=1,
        baselineContrast=0.5,
        interProbeIntervalRange=(5, 10),
        trialCount=3,
        cooldownPeriod=3,
        ):

        #
        clock = core.MonotonicClock()
        self.metadata = list()

        cpp = spatialFrequency / self.display.ppd # cycles per pixel
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
            sf=cpp,
            contrast=baselineContrast,
        )

        #
        gabor.draw()
        self.display.flip()
        trialNumber = 1
        for trialIndex in range(trialCount):
            interProbeInterval = np.random.randint(
                round(interProbeIntervalRange[0] * self.display.fps),
                round(interProbeIntervalRange[1] * self.display.fps),
                1
            ).item()
            for frameIndex in range(interProbeInterval):
                gabor.draw()
                self.display.flip()
            gabor.contrast = probeContrast
            self.display.flashSignalPatch(3)
            self.metadata.append([trialNumber, clock.getTime()])
            for frameIndex in range(round(probeDuration * self.display.fps)):
                gabor.draw()
                self.display.flip()
            gabor.contrast = baselineContrast
            trialNumber += 1

        #
        for frameIndex in range(round(cooldownPeriod * self.display.fps)):
            gabor.draw()
            self.display.flip()

        #
        self.display.clearStimuli()
        self.metadata = np.array(self.metadata)

        return None

    def saveMetadata(self, sessionFolder):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)

        if self.metadata is None:
            return

        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        filename = str(sessionFolderPath.joinpath('staticGratingWithProbeMetadata.txt'))
        with open(filename, 'w') as stream:
            stream.write(f'Trial, Timestamp\n')
            for trialNumber, timestamp in self.metadata:
                stream.write(f'{trialNumber:.0f}, {timestamp:.3f}\n')

        # Delete metadata
        self.metadata = None

        return
