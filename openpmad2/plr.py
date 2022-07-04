import math
import numpy as np
import pathlib as pl
from psychopy import visual
from itertools import product
from myphdlib import labjack as lj

def postprocess(sessionFolder, camera=6, stimulus=7):
    """
    """

    sessionFolderPath = pl.Path(sessionFolder)

    #
    results = list(sessionFolderPath.rglob('*labjack')) + list(sessionFolderPath.rglob('*LabJack'))
    if len(results) != 1:
        raise Exception('Could not located LabJack folder')
    labjackFolder = str(results.pop())

    #
    data = lj.loadLabjackData(labjackFolder)
    timestamps = data[:, 0]
    exposureOnsetSignal, exposureOnsetIndices = lj.extractLabjackEvent(data, camera, edge='both')
    stimulusOnsetSignal, stimulusOnsetIndices = lj.extractLabjackEvent(data, stimulus, edge='rising')

    # Discared the first stimulus; this is the dark adaptation period
    stimulusOnsetIndices = stimulusOnsetIndices[1:]

    #
    stimulusOnsetTimestampsFilePath = str(sessionFolderPath.joinpath('stimulusOnsetTimestamps.txt'))
    with open(stimulusOnsetTimestampsFilePath, 'w') as stream:
        for timestamp in timestamps[stimulusOnsetIndices][0::2]:
            stream.write(f'{timestamp:.3f}\n')
    
    stimulusOffsetTimestampsFilePath = str(sessionFolderPath.joinpath('stimulusOffsetTimestamps.txt'))
    with open(stimulusOffsetTimestampsFilePath, 'w') as stream:
        for timestamp in timestamps[stimulusOnsetIndices][1::2]:
            stream.write(f'{timestamp:.3f}\n')
    
    exposureOnsetTimestampsFilePath = str(sessionFolderPath.joinpath('exposureOsetTimestamps.txt'))
    with open(exposureOnsetTimestampsFilePath, 'w') as stream:
        for timestamp in timestamps[exposureOnsetIndices]:
            stream.write(f'{timestamp:.3f}\n')  

    return

class LightSteps():
    """
    """

    def __init__(self, display=None):
        """
        """

        self.display = display
        self.metadata = None

        return

    def present(
        self,
        lightLevels=np.around(np.logspace(math.log(0.2, 10), 0, 5), 2) ,
        trialDuration=1,
        itiDuration=1,
        repeats=3,
        darkAdaptationPeriod=3,
        colors=('r', 'b'),
        ):
        """
        """

        #
        lightLevelsScaled = np.interp(lightLevels, (0, 1), (-1, 1))
        combos = np.array(list(product(lightLevels, colors)))
        self.metadata = np.tile(combos, repeats).reshape(len(combos) * repeats, 2)
        np.random.shuffle(self.metadata)

        #
        patch = visual.Rect(self.display, size=self.display.size, units=self.display.units)

        # Dark adaptation period (same duration as ISI)
        patch.setColor([-1, -1, -1], colorSpace='rgb')
        self.display.flashSignalPatch(3)
        for frameIndex in range(round(self.display.fps * darkAdaptationPeriod)):
            patch.draw()
            self.display.flip()

        #
        for lightLevel, color in self.metadata:
            lightLevelScaled = np.interp(lightLevel, (0, 1), (-1, 1))
            if color == 'r':
                value = [lightLevelScaled, -1, -1]
            elif color == 'b':
                value = [-1, -1, lightLevelScaled]
            patch.setColor(value, colorSpace='rgb')
            self.display.flashSignalPatch(3)
            for frameIndex in range(round(self.display.fps * trialDuration)):
                patch.draw()
                self.display.flip()
            patch.setColor([-1, -1, -1], colorSpace='rgb')
            self.display.flashSignalPatch(3)
            for frameIndex in range(round(self.display.fps * itiDuration)):
                patch.draw()
                self.display.flip()

        self.display.clearStimuli()

        return None

    def saveMetadata(self, sessionFolder):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)


        if self.metadata is None:
            return

        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        filename = str(sessionFolderPath.joinpath('lightStepsMetadata.txt'))
        with open(filename, 'w') as stream:
            stream.write('Trial (1-n), Intensity (0-1), color (r/b)\n')
            for trialIndex, (lightLevel, color) in enumerate(self.metadata):
                trialNumber = trialIndex + 1
                stream.write(f'{trialNumber}, {float(lightLevel):.2f}, {color}\n')

        return
