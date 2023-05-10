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

                # TODO: Figure out why the phase needs to be multiplied by -1???
                gabor.phase += (phase * motion * -1)
                gabor.contrast = contrast

                #
                if signal:
                    self.display.signalEvent(2, units='frames')
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

class DriftingGratingWithFictiveSaccades2():
    """
    """

    def __init__(self, display):
        """
        """

        self._eventIndex = 0
        self.metadata = None
        self.display = display

        return

    def _computeGratingPhaseAcrossSaccade(
        self,
        saccadeVelocityParams,
        saccadeDurationInFrames,
        gratingVelocity,
        spatialFrequency,
        ):
        """
        """

        peak, width, constant = saccadeVelocityParams
        if constant:
            pass

        #
        if constant:
            sequence = np.full(saccadeDurationInFrames + 2, peak)

        #
        else:
            x0 = (saccadeDurationInFrames + 2) // 2
            x = np.arange(saccadeDurationInFrames + 2)
            v = (np.exp(-np.power(x - x0, 2.) / (2 * np.power(width, 2.))))
            sequence = np.interp(v, (v.min(), v.max()), (gratingVelocity, peak))
        
        #
        cpf = spatialFrequency * sequence / self.display.fps

        return cpf[1:-1]

    def _generateMetadata(
        self,
        nTrialsPerCondition,
        gratingMotion,
        randomizeTrialOrder
        ):
        """
        """

        self.metadata = {
            'trials': list(),
        }

        #
        for motion, block in zip(gratingMotion, range(len(gratingMotion))):
            trials = list()
            for trialType in ('saccade', 'probe', 'combined'):
                for iTrial in range(nTrialsPerCondition):
                    trial = (block, motion, trialType)
                    trials.append(trial)
            index = np.arange(len(trials))
            if randomizeTrialOrder:
                index = np.random.choice(index, size=index.size, replace=False)
            for iTrial in index:
                self.metadata['trials'].append(trials[iTrial])

        #
        nTrials = len(self.metadata['trials'])
        nEvents = int(nTrials * 2)
        self.metadata['events'] = np.full([nEvents, 1], '').astype(object)

        return

    def _runMainLoop(
        self,
        spatialFrequency,
        gratingVelocity,
        saccadeDurationInFrames,
        saccadeVelocityParams,
        baselineContrast,
        probeDurationInFrames,
        probeLatencyInFrames,
        probeContrast,
        itiRange,
        tStatic,
        tWarmup,
        ibi
        ):
        """
        """

        #
        cpp = spatialFrequency / self.display.ppd # cycles per pixel
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
            sf=cpp,
        )
        gabor.contrast = baselineContrast

        #
        cpf = spatialFrequency * gratingVelocity / self.display.fps
        phases = self._computeGratingPhaseAcrossSaccade(
            saccadeVelocityParams,
            saccadeDurationInFrames,
            gratingVelocity,
            spatialFrequency
        )

        # Warm-up
        motion = self.metadata['trials'][0][1]
        for iFrame in range(int(round(ibi * self.display.fps))):
            self.display.drawBackground()
            self.display.flip()
        for iFrame in range(int(round(tStatic * self.display.fps))):
            gabor.draw()
            self.display.flip()
        for iFrame in range(int(round(tWarmup * self.display.fps))):
            gabor.phase += cpf * motion
            gabor.draw()
            self.display.flip()

        #
        iEvent = 0
        for block, motion_, tt in self.metadata['trials']:

            # Block transition
            if motion_ != motion:
                motion = motion_
                for iFrame in range(int(round(ibi * self.display.fps))):
                    self.display.drawBackground()
                    self.display.flip()
                for iFrame in range(int(round(tStatic * self.display.fps))):
                    gabor.draw()
                    self.display.flip()
                for iFrame in range(int(round(tWarmup * self.display.fps))):
                    gabor.phase += cpf * motion
                    gabor.draw()
                    self.display.flip()

            # Saccade-only trials
            if tt == 'saccade':
                for iFrame, phase in enumerate(phases):
                    if iFrame == 0:
                        self.display.signalEvent(2, units='frames')
                        self.metadata['events'][iEvent] = 'saccade onset'
                        iEvent += 1
                    gabor.phase += phase * motion
                    gabor.draw()
                    self.display.flip()

            # Probe-only trials
            elif tt == 'probe':
                gabor.contrast = probeContrast
                for iFrame in range(probeDurationInFrames):
                    if iFrame == 0:
                        self.display.signalEvent(2, units='frames')
                        self.metadata['events'][iEvent] = 'probe onset'
                        iEvent += 1
                    gabor.phase += cpf * motion
                    gabor.draw()
                    self.display.flip()
                gabor.contrast = baselineContrast

            # Saccade and probe trials
            elif tt == 'combined':
                probeOffsetCountdown = 0
                for iFrame, phase in enumerate(phases):
                    if iFrame == 0:
                        self.display.signalEvent(1, units='frames')
                        self.metadata['events'][iEvent] = 'saccade onset'
                        iEvent += 1
                    if probeOffsetCountdown == 0 and gabor.contrast != baselineContrast:
                        gabor.contrast = baselineContrast
                    gabor.phase += phase * motion
                    if iFrame == probeLatencyInFrames:
                        self.display.signalEvent(1, units='frames')
                        self.metadata['events'][iEvent] = 'probe onset'
                        iEvent += 1
                        probeOffsetCountdown = probeDurationInFrames
                        gabor.contrast = probeContrast
                    gabor.draw()
                    self.display.flip()
                    # print(iFrame, gabor.contrast, self.display.state)
                    if probeOffsetCountdown != 0:   
                        probeOffsetCountdown -= 1
                if gabor.contrast != baselineContrast:
                    while probeOffsetCountdown != 0:
                        gabor.phase += cpf * motion
                        gabor.draw()
                        self.display.flip()
                    gabor.contrast = baselineContrast

            # ITI
            itiInSeconds = np.random.uniform(low=itiRange[0], high=itiRange[1], size=1).item()
            itiInFrames = int(round(self.display.fps * itiInSeconds))
            for iFrame in range(itiInFrames):
                gabor.phase += cpf * motion
                gabor.draw()
                self.display.flip()

        #
        for iFrame in range(int(round(ibi * self.display.fps))):
            self.display.drawBackground()
            self.display.flip()

        return

    def present(
        self,
        nTrialsPerCondition=1,
        spatialFrequency=0.15,
        gratingVelocity=12,
        saccadeDurationInFrames=7,
        probeContrast=1,
        baselineContrast=0.5,
        probeLatencyInFrames=2,
        probeDurationInFrames=3,
        saccadeVelocityParams=(300, 1, 0), # Peak, width (SD), and flag for constant velocity
        gratingMotion=(-1, 1),
        randomizeTrialOrder=True,
        itiRange=(1, 2),
        tStatic=3,
        tWarmup=3,
        ibi=3,
        ):
        """
        """

        self._generateMetadata(
            nTrialsPerCondition,
            gratingMotion,
            randomizeTrialOrder
        )

        self._runMainLoop(
            spatialFrequency,
            gratingVelocity,
            saccadeDurationInFrames,
            saccadeVelocityParams,
            baselineContrast,
            probeDurationInFrames,
            probeLatencyInFrames,
            probeContrast,
            itiRange,
            tStatic,
            tWarmup,
            ibi
        )

        return

    def saveMetadata(self, sessionFolder):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        #
        index = np.array([len(element.item()) > 0 for element in self.metadata['events']])
        self.metadata['events'] = self.metadata['events'][index, :]

        with open(sessionFolderPath.joinpath('fictiveSaccadeMetadata.pkl'), 'wb') as stream:
            pickle.dump(self.metadata, stream)

        return