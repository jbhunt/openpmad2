import copy
import pickle
import numpy as np
import pathlib as pl
from psychopy import visual

class DriftingGratingWithFictiveSaccades():
    """
    """

    def __init__(self, display):
        """
        """

        self._eventIndex = 0
        self.metadata = None
        self.display = display

        return

    def _generateMetadata(
        self,
        nTrialsPerBlock,
        nBlocksPerDirection,
        gratingMotion,
        randomizeBlocks,
        ):
        """
        """

        #
        nTrialsTotal = len(gratingMotion) * nBlocksPerDirection * nTrialsPerBlock * 2

        #
        self.metadata = {
            'blocks' : np.full([nTrialsTotal, 1], 0),
            'motion': np.full([nTrialsTotal, 1], np.nan),
            'probed': np.full([nTrialsTotal, 1], False),
            'events': np.full([1000000, 1], '').astype(object),
            'timestamps': np.full([1000000, 1], np.nan)
        }

        #
        iTrial = 0
        iBlock = 0
        for motion in gratingMotion:
            for iBlock_ in range(nBlocksPerDirection):
                for iTrial_ in range(nTrialsPerBlock):
                    for probed in (True, False):
                        self.metadata['blocks'][iTrial] = iBlock + 1
                        self.metadata['motion'][iTrial] = motion
                        self.metadata['probed'][iTrial] = probed
                        iTrial += 1
                iBlock += 1

        # Randomize blocks
        if randomizeBlocks:

            #
            uniqueBlockIndices = np.unique(self.metadata['blocks'].flatten())
            uniqueBlockIndices.sort()
            shuffledBlockIndices = np.copy(uniqueBlockIndices)

            #
            np.random.shuffle(shuffledBlockIndices)
            metadata2 = copy.deepcopy(self.metadata)

            #
            for iBlockTarget, iBlockSource in enumerate(shuffledBlockIndices):
                rowIndicesSource, columnIndicesSource = np.where(self.metadata['blocks'] == iBlockSource)
                rowIndicesTarget = np.arange(nTrialsPerBlock * 2) + (iBlockTarget * nTrialsPerBlock * 2)
                for key in ('blocks', 'motion', 'probed'):
                    try:
                        metadata2[key][rowIndicesTarget] = self.metadata[key][rowIndicesSource]
                    except:
                        import pdb; pdb.set_trace()
                    
            #
            self.metadata = None
            self.metadata = metadata2

        return
    
    def _computeGratingPhaseDuringSaccades(
        self,
        nFramesPerSaccade,
        gaussianFunctionParams,
        gratingVelocity,
        spatialFrequency,
        constant=False
        ):
        """
        Compute the velocity profile of the main sequence
        """

        #
        peak, width = gaussianFunctionParams

        #
        if constant:
            sequence = np.full(nFramesPerSaccade + 2, peak)

        #
        else:
            x0 = (nFramesPerSaccade + 2) // 2
            x = np.arange(nFramesPerSaccade + 2)
            v = (np.exp(-np.power(x - x0, 2.) / (2 * np.power(width, 2.))))
            sequence = np.interp(v, (v.min(), v.max()), (gratingVelocity, peak))
        
        #
        cpf = spatialFrequency * sequence / self.display.fps

        return cpf[1:-1]
    
    def _generateSaccadeSequence(
        self,
        probeContrast,
        baselineContrast,
        probeLatencyInSeconds,
        probeDuration,
        cpf1,
        cpf2,
        nFramesPerSaccade,
        ):
        """
        """

        # Pre-determine the contrast and phase for each frame
        probeLatencyInFrames = int(round(self.display.fps * probeLatencyInSeconds))
        probeDurationInFrames = int(round(self.display.fps * probeDuration))
        nFramesInSequence = probeLatencyInFrames + probeDurationInFrames + 1

        #
        sequence = {
            'probed': {
                'phase': np.full(nFramesInSequence, np.nan),
                'signal': np.full(nFramesInSequence, False),
                'contrast': np.full(nFramesInSequence, np.nan)
            },
            'unprobed': {                
                'phase': np.full(nFramesInSequence, np.nan),
                'signal': np.full(nFramesInSequence, False),
                'contrast': np.full(nFramesInSequence, np.nan)
            },
        }

        #
        for iFrame in range(nFramesInSequence):

            #
            if iFrame == 0:
                sequence['probed']['signal'][iFrame] = True
                sequence['unprobed']['signal'][iFrame] = True
            elif iFrame == probeLatencyInFrames:
                sequence['probed']['signal'][iFrame] = True
                sequence['unprobed']['signal'][iFrame] = False
            else:
                sequence['probed']['signal'][iFrame] = False

            #
            if iFrame < nFramesPerSaccade:
                sequence['probed']['phase'][iFrame] = cpf2[iFrame]
                sequence['unprobed']['phase'][iFrame] = cpf2[iFrame]
            else:
                sequence['probed']['phase'][iFrame] = cpf1
                sequence['unprobed']['phase'][iFrame] = cpf1

            # Contrast
            if iFrame >= probeLatencyInFrames:
                sequence['probed']['contrast'][iFrame] = probeContrast
                sequence['unprobed']['contrast'][iFrame] = baselineContrast
            else:
                sequence['probed']['contrast'][iFrame] = baselineContrast
                sequence['unprobed']['contrast'][iFrame] = baselineContrast

        # Last frame in the sequence returns the contrast to baseline
        for key in ('probed', 'unprobed'):
            for param, value in zip(['phase', 'signal', 'contrast'], [cpf1, False, baselineContrast]):
                sequence[key][param][-1] = value

        return sequence
    
    def _runMainLoop(
        self,
        gabor,
        spatialFrequency,
        gratingVelocity,
        gaussianFunctionParams,
        nFramesPerSaccade,
        baselineContrast,
        probeContrast,
        probeLatencyInSeconds,
        probeDuration,
        interSaccadeIntervalRange,
        tIdle,
        tMargin,
        tStatic,
        tIBI,
        constantSaccadeVelocity
        ):
        """
        """

        #
        cpf1 = spatialFrequency * gratingVelocity / self.display.fps
        cpf2 = self._computeGratingPhaseDuringSaccades(
            nFramesPerSaccade,
            gaussianFunctionParams,
            gratingVelocity,
            spatialFrequency,
            constantSaccadeVelocity
        )
        gabor.contrast = baselineContrast

        #
        fictiveSaccadeSequence = self._generateSaccadeSequence(
            probeContrast,
            baselineContrast,
            probeLatencyInSeconds,
            probeDuration,
            cpf1,
            cpf2,
            nFramesPerSaccade,
        )

        #
        blockTransitionIndices = np.where(
            np.diff(self.metadata['blocks'].flatten()) != 0
        )[0] + 1
        blockTransitionIndices = np.concatenate([
            blockTransitionIndices,
            np.array([0])
        ])
        blockTransitionIndices.sort()

        #
        iterable = zip(
            self.metadata['blocks'],
            self.metadata['motion'],
            self.metadata['probed'],
        )

        #
        if self.display.backgroundColor != 0:
            self.display.setBackgroundColor(0)
        for iFrame in range(int(np.ceil(self.display.fps * tIdle))):
            self.display.drawBackground()
            self.display.flip()

        #
        iEvent = 0
        for iTrial, (iBlock, motion, probed) in enumerate(iterable):
            
            # Transition to new block
            if iTrial in blockTransitionIndices:

                # Show a blank screen
                for iFrame in range(int(np.ceil(self.display.fps * tIBI))):
                    self.display.drawBackground()
                    self.display.flip()

                # Show the grating static
                for iFrame in range(int(np.ceil(self.display.fps * tStatic))):
                    gabor.draw()
                    self.display.flip()

                # Show the grating in motion
                for iFrame in range(int(np.ceil(self.display.fps * tMargin))):
                    gabor.phase += (cpf1 * motion)
                    gabor.draw()
                    self.display.flip()

            # Choose an inter trial interval
            iti = np.random.uniform(
                low=interSaccadeIntervalRange[0],
                high=interSaccadeIntervalRange[1],
                size=1
            )
            for iFrame in range(int(np.ceil(self.display.fps * iti))):
                gabor.phase += (cpf1 * motion)
                gabor.draw()
                self.display.flip()

            # Present the fictive saccade
            key = 'probed' if probed else 'unprobed'
            recordTimestamp = False
            for iFrame, (phase, signal, contrast) in enumerate(zip(*fictiveSaccadeSequence[key].values())):

                #
                gabor.phase += (phase * motion)
                gabor.contrast = contrast

                #
                if signal:
                    self.display.signalEvent(1, units='frames')
                    if iFrame == 0:
                        event = 'saccade onset'
                    else:
                        event = 'probe onset'
                    recordTimestamp = True
                gabor.draw()
                timestamp = self.display.flip()

                #
                if recordTimestamp:
                    self.metadata['events'][iEvent] = event
                    self.metadata['timestamps'][iEvent] = timestamp
                    iEvent += 1
                    recordTimestamp = False

            #
            if iTrial + 1 in blockTransitionIndices:
                for iFrame in range(int(np.ceil(self.display.fps * tMargin))):
                    gabor.phase += (cpf1 * motion)
                    gabor.draw()
                    self.display.flip()

        for iFrame in range(int(np.ceil(self.display.fps * tIdle))):
            self.display.drawBackground()
            self.display.flip()

        #
        mask = np.array([True if len(entry.item()) != 0 else False for entry in self.metadata['events']])
        self.metadata['events'] = self.metadata['events'][mask, :]
        self.metadata['timestamps'] = self.metadata['timestamps'][mask, :]

        return

    def present(
        self,
        spatialFrequency=0.15,
        gratingVelocity=12,
        nFramesPerSaccade=5,
        gaussianFunctionParams=(300, 1), # Peak and width
        baselineContrast=0.4,
        probeContrast=1,
        probeLatencyInSeconds=0.05,
        probeDuration=0.05,
        interSaccadeIntervalRange=(0.5, 1),
        nTrialsPerBlock=1,
        nBlocksPerDirection=1,
        gratingMotion=(-1, 1),
        tMargin=1,
        tStatic=1,
        tIBI=1,
        tIdle=1,
        randomizeBlocks=True,
        constantSaccadeVelocity=False
        ):
        """
        """

        #
        self._eventIndex = 0

        #
        cpp = spatialFrequency / self.display.ppd # cycles per pixel
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
            sf=cpp,
        )

        #
        self._generateMetadata(
            nTrialsPerBlock,
            nBlocksPerDirection,
            gratingMotion,
            randomizeBlocks,  
        )

        #
        self._runMainLoop(
            gabor,
            spatialFrequency,
            gratingVelocity,
            gaussianFunctionParams,
            nFramesPerSaccade,
            baselineContrast,
            probeContrast,
            probeLatencyInSeconds,
            probeDuration,
            interSaccadeIntervalRange,
            tIdle,
            tMargin,
            tStatic,
            tIBI,
            constantSaccadeVelocity
        )

        return
    
    def saveMetadata(self, sessionFolder):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        with open(sessionFolderPath.joinpath('fictiveSaccadeMetadata.pkl'), 'wb') as stream:
            pickle.dump(self.metadata, stream)

        return