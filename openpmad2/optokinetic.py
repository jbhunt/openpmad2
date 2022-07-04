import numpy as np
import pathlib as pl
from itertools import product
from psychopy import visual
from decimal import Decimal
from .constants import N_SIGNAL_FRAMES
from .constants import CLOCKWISE_MOTION, COUNTER_CLOCKWISE_MOTION

def getStimulusParameters(f1, fn, v1, vn, c1, cn, repeats=1, shuffleTrials=True, motionDirection=['cw', 'ccw']):
    """
    """

    stimulusParameters = list()
    for param in ['frequency', 'velocity', 'contrast']:
        if param == 'frequency':
            f =  fn
            v = [v1]
            c = [c1]
        elif param == 'velocity':
            f = [f1]
            v =  vn
            c = [c1]
        elif param == 'contrast':
            f = [f1]
            v = [v1]
            c =  cn

        #
        for combo in product(f, v, c):
            for irep in range(repeats):
                if combo not in stimulusParameters:
                    stimulusParameters.append(combo)

    stimulusParameters = np.vstack([
        np.array(stimulusParameters),
        np.array(stimulusParameters)
    ])
    motionDirection = np.vstack([
        np.full([int(stimulusParameters.shape[0] / 2), 1],  1),
        np.full([int(stimulusParameters.shape[0] / 2), 1], -1)
    ])
    stimulusParameters = np.hstack([stimulusParameters, motionDirection])

    #
    stimulusParameters = np.repeat(stimulusParameters, repeats, axis=0)

    if shuffleTrials:
        np.random.shuffle(stimulusParameters)

    return stimulusParameters

class OptokineticDrum():
    """
    """

    def __init__(self, display=None):
        self.display = display
        self.metadata = None
        return

    def present(
        self,
        bestFrequency    =  0.2,
        bestVelocity     =  12,
        bestContrast     =  1,
        testFrequencySet = [0.05 , 0.1, 0.2, 0.3, 0.5],
        testVelocitySet  = [5, 10, 15 , 25           ],
        testContrastSet  = [0.05 , 0.1, 0.2, 0.5, 1  ],
        motionDirection  = ['cw', 'ccw'],
        motionDuration   =  30,
        staticDuration   =  0.05,
        isiDuration      =  3,
        repeats          =  3,
        shuffleTrials    =  True,
        ):
        """
        """

        # Determine the combination of frequency, velocity, and contrast
        self.metadata = getStimulusParameters(
            bestFrequency,
            testFrequencySet,
            bestVelocity,
            testVelocitySet,
            bestContrast,
            testContrastSet,
            repeats,
            shuffleTrials,
            motionDirection,
        )

        #
        totalTimeEstimate = 0
        for itrial in range(len(self.metadata)):
            totalTimeEstimate += (staticDuration + motionDuration + isiDuration) / 60
        print(f'Estimated stimulus duration: {totalTimeEstimate:.2f} minutes')

        #
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
        )

        #
        for frequency, velocity, contrast, direction in self.metadata:

            # print(f'f={frequency:.2f}, v={velocity:.2f}, c={contrast:.2f}, direction={direction}')

            #
            gabor.sf = frequency / self.display.ppd # cycles per pixel
            gabor.contrast = contrast # Contrast (0 to 1)
            cpf = float(
                Decimal(str(frequency)) * Decimal(str(velocity)) / Decimal(str(self.display.fps)) # cycles per frame
            )

            #
            for iframe in range(int(np.ceil(self.display.fps * staticDuration))):
                gabor.draw()
                self.display.flip()

            #
            self.display.state = True
            for iframe in range(int(np.ceil(self.display.fps * motionDuration))):
                if iframe == N_SIGNAL_FRAMES:
                    self.display.state = False
                gabor.phase += cpf * direction
                gabor.draw()
                self.display.flip()

            #
            self.display.state = True
            for iframe in range(int(np.ceil(self.display.fps * isiDuration))):
                if iframe == N_SIGNAL_FRAMES:
                    self.display.state = False
                self.display._background.draw()
                self.display.flip()

        return

    def saveMetadata(self, dstFolder):
        """
        """

        if self.metadata is None:
            return

        dstFolderPath = pl.Path(dstFolder)
        if dstFolderPath.exists() is False:
            return

        filename = str(dstFolderPath.joinpath(f'driftingGratingMetadata.txt'))
        with open(filename, 'w') as stream:
            for frequency, velocity, contrast, direction in self.metadata:
                stream.write(f'{frequency:.2f}, {velocity:.2f}, {contrast:.2f}, {direction}\n')

        return
