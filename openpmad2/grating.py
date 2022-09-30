import numpy as np
import pathlib as pl
from itertools import product
from psychopy.visual import GratingStim
from datetime import datetime as dt
from . import bases

class DirectionSelectivityProtocol(bases.StimulusBase):
    """
    """

    def __init__(self, display):
        """
        """
        super().__init__(display)
        self.metadata = None
        self._temporalFrequency = None
        return

    def present(
        self,
        orientations=np.linspace(0, 360, 9),
        spatialFrequencies=np.logspace(np.log10(0.01), np.log10(0.32), 6),
        temporalFrequency=2,
        stimulusDuration=2,
        itiDuration=1,
        repeats=1,
        warmupDuration=5,
        ):
        """
        """

        self._temporalFrequency = temporalFrequency

        #
        hypotenuse = np.sqrt(self.display.width ** 2 + self.display.width ** 2)
        gabor = GratingStim(self.display, units='pix', size=hypotenuse)

        #
        N = len(spatialFrequencies) * len(orientations) * 2 * repeats
        self.metadata = np.full([N, 3], np.nan)
        trialIndex = 0
        for i in range(repeats):
            block = np.array(list(product(spatialFrequencies, orientations, [-1, 1])))
            np.random.shuffle(block)
            for entry in block:
                self.metadata[trialIndex, :] = entry
                trialIndex += 1

        # How long will the stimulus last?
        totalStimulusTime = N * (stimulusDuration + itiDuration) + warmupDuration
        totalStimulusTime /= 60
        # print(f'Stimulus duration estimate: ~{totalStimulusTime:.2f} minutes')

        # Warmup period
        self.display.idle(warmupDuration, units='seconds')

        # Main loop
        for spatialFrequency, orientation, direction in self.metadata:

            #
            cpp = spatialFrequency / self.display.ppd
            dps = temporalFrequency / spatialFrequency
            cpf = spatialFrequency * dps / self.display.fps

            #
            gabor.sf = cpp
            gabor.ori = orientation

            #
            self.display.signalEvent(3, units='frames')
            for frameIndex in range(round(self.display.fps * stimulusDuration)):
                gabor.phase += cpf * direction
                gabor.draw()
                self.display.flip()

            #
            self.display.signalEvent(3, units='frames')
            self.display.idle(itiDuration, units='seconds')

        return

    def saveMetadata(self, sessionFolder, headerSize=2):
        """
        """

        sessionFolderPath = super().saveMetadata(sessionFolder)
        fullFilePath = sessionFolderPath.joinpath(f'directionalGratingMetadata.txt')

        #
        lines = None
        if fullFilePath.exists():
            with open(fullFilePath, 'r') as stream:
                lines = stream.readlines()

        with open(fullFilePath, 'w') as stream:
            if self._temporalFrequency is None:
                value = '?'
            else:
                value = self._temporalFrequency
            stream.write(f'Temporal frequency={value} Hz\n')
            stream.write(f'Spatial frequency (cycles/degree), Orientation (degrees), Direction (-1, 1)\n')
            if lines is not None:
                for line in lines[headerSize:]:
                    stream.write(line)
            for spatialFrequency, orientation, direction in self.metadata:
                line = f'{spatialFrequency:.2f}, {orientation:.0f}, {direction:.0f}\n'
                stream.write(line)

        return