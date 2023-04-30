from . import bases
from openpmad2.helpers import estimateFrameCount, generateMetadataFilename
import time
import pickle
import numpy as np
import pathlib as pl

class PerformanceBenchmarkingStimulus(bases.StimulusBase):
    """
    """

    def _generateMetadata(
        self,
        nFramesLow,
        nFramesHigh,
        nFramesIdle,
        nCycles,
        ):
        """
        """

        nFramesTotal = (2 * nFramesIdle) + (nCycles * (nFramesLow + nFramesHigh))
        self.metadata = {
            'timestamps': np.full([nFramesTotal, 1], np.nan)
        }

        return

    def present(
        self,
        tLow=1/60,
        tHigh=1/60,
        nCycles=20,
        tIdle=1,
        backgroundColor=0
        ):
        """
        """

        #
        self.display.setBackgroundColor(backgroundColor)

        #
        nFramesLow = estimateFrameCount(tLow, self.display.fps)
        nFramesHigh = estimateFrameCount(tHigh, self.display.fps)
        nFramesIdle = estimateFrameCount(tIdle, self.display.fps)
        self._generateMetadata(
            nFramesLow,
            nFramesHigh,
            nFramesIdle,
            nCycles,
        )

        #
        iFrame = 0
        for iFrame_ in range(nFramesIdle):
            self.display.drawBackground()
            self.metadata['timestamps'][iFrame] = self.display.flip()
            iFrame += 1

        #
        for iCycle in range(nCycles):
            self.display.state = True
            for iFrame_ in range(nFramesHigh):
                self.display.drawBackground()
                self.metadata['timestamps'][iFrame] = self.display.flip()
                iFrame += 1
            self.display.state = False
            for iFrame_ in range(nFramesLow):
                self.display.drawBackground()
                self.metadata['timestamps'][iFrame] = self.display.flip()
                iFrame += 1
        
        #
        for iFrame_ in range(nFramesIdle):
            self.display.drawBackground()
            self.metadata['timestamps'][iFrame] = self.display.flip()
            iFrame += 1

        return

    def saveMetadata(self, sessionFolder):
        """
        """

        #
        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        # Save the metadata dict
        filename = generateMetadataFilename(
            sessionFolderPath,
            'bencharkingStimulusMetadata',
            '.pkl'
        )
        with open(filename, 'wb') as stream:
            pickle.dump(self.metadata, stream)

        return
