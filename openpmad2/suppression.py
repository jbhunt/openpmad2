import copy
import multiprocessing as mp
import pathlib as pl
from email.policy import default

import numpy as np
from psychopy import visual

from . import bases
from myphdlib.general.toolkit import smooth


class StateManager():
    """
    """

    def __init__(self):
        """
        """

        #
        self.defaultPathIndex = 0
        self.perisaccadicProbesPathIndex = 0
        self.fictiveSaccadesPathIndex = 1
        self.randomProbesPathIndex = 2
        self.realtimeProbesOnlyPathIndex = -1

        #
        self.paths = np.array([
            [0, 1, 2, 3], # Present perisaccadic probes
            [0, 4, 3],    # Present fictive saccades
            [0, 1, 5, 3], # Present random probes
            [0]           #
        ], dtype='object')
        self.ipath = 0
        self.istate = 0

        #
        self._finalState = self.paths[self.ipath][-1]

        #
        self._inTimeout = False
        self._inInterEventInterval = True
        self._inForePeriod = False
        self._presentingPerisaccadicProbe = False
        self._inRefractoryPeriod = False
        self._presentingFictiveSaccade = False
        self._presentingRandomProbe = False

        #
        self._states = {
            0: self._inInterEventInterval,
            1: self._inForePeriod,
            2: self._presentingPerisaccadicProbe,
            3: self._inRefractoryPeriod,
            4: self._presentingFictiveSaccade,
            5: self._presentingRandomProbe
        }

        self._score = list()

        return

    def changeState(self, ipath=0):
        """
        """

        #
        oldState = self.paths[self.ipath][self.istate]
        if oldState == self._finalState:
            self.istate = 0
            self.ipath = 0
        else:
            self.istate += 1

        #
        self.ipath = ipath

        #
        newState = self.paths[self.ipath][self.istate]

        # Reset state values and assign new state
        for key in self._states.keys():
            self._states[key] = False
        self._states[newState] = True

        return

    def choosePath(self, presentRandomProbes=False, presentFictiveSaccades=False):
        """
        """

        if presentRandomProbes == True and presentFictiveSaccades == True:
            pathIndex = np.random.choice([1, 2], size=1).item()
        elif presentRandomProbes == True and presentFictiveSaccades == False:
            pathIndex = 1
        elif presentRandomProbes == False and presentFictiveSaccades == True:
            pathIndex = 2

        return pathIndex

    def recordStates(self):
        """
        """

        snapshot = np.array(list(self._states.values()))
        self._score.append(snapshot)
        return

    #
    @property
    def score(self):
        return np.array(self._score)

    # 1
    @property
    def inInterEventInterval(self):
        return self._states[0]

    # 2
    @property
    def inForeperiod(self):
        return self._states[1]

    # 3
    @property
    def presentingPerisaccadicProbe(self):
        return self._states[2]

    # 4
    @property
    def inRefractoryPeriod(self):
        return self._states[3]

    # 5
    @property
    def presentingFictiveSaccade(self):
        return self._states[4]

    # 6
    @property
    def presentingRandomProbe(self):
        return self._states[5]

    @property
    def state(self):
        currentState = np.array([
            key for key in self._states
                if self._states[key] is True
        ]).item()
        return currentState

class DriftingGratingWithRealTimeProbe():
    """
    """

    def __init__(
        self,
        display,
        shared=None,
        frequency=0.15,
        velocity=12,
        tstatic=5,
        ntrials=1,
        duration=3,
        iti=5,
        directions=(-1, 1),
        randomize=True,
        ):
        """
        """

        if shared == None:
            self.shared = mp.Value('i', 0)
        else:
            self.shared = shared

        self.display = display
        self.frequency = frequency
        self.velocity = velocity
        self.tstatic = tstatic
        self.ntrials = ntrials
        self.duration = duration
        self.iti = iti
        self.order = np.tile(directions, self.ntrials)
        if randomize:
            np.random.shuffle(self.order)

        self.metadata = None

        return

    def present(
        self,
        warmup=1,
        timeout=3,
        tprobe=0.05,
        tmargin=3,
        isirange=(0.5, 1),
        presentRandomProbes=True,
        presentFictiveSaccades=False,
        fictiveSaccadeDuration=0.06,
        fictiveSaccadeVelocity=300,
        foreperiodSample=None,
        baselineContrastLevel=0.5,
        returnStateValues=False,
        defaultMetadataSize=10000,
        ):
        """
        """

        #
        metadata = np.full(defaultMetadataSize, '', dtype=object)

        #
        cpp = self.frequency / self.display.ppd # cycles per pixel
        cpf1 = self.frequency * self.velocity / self.display.fps
        cpf2 = self.frequency * fictiveSaccadeVelocity / self.display.fps

        #
        frameIndexProbesAllowed = round(self.display.fps * tmargin) - 1
        frameIndexProbesDisallowed = round(self.display.fps * self.duration) - round(self.display.fps * tmargin) - 1

        #
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
            sf=cpp,
            contrast=baselineContrastLevel,
        )

        #
        image = np.full([self.display.height, self.display.width], 0)
        background = visual.ImageStim(
            self.display,
            image=image,
            size=self.display.size,
            units='pix',
        )

        # Timestamps
        ntrials = self.order.size
        nframes = (
            int(np.ceil(self.display.fps * warmup))                  +
            int(np.ceil(self.display.fps * self.tstatic) * ntrials)  +
            int(np.ceil(self.display.fps * self.duration) * ntrials) +
            int(np.ceil(self.display.fps * self.iti) * ntrials)
        )
        timestamps = np.full(nframes, np.nan)
        eventTimestamps = np.full(defaultMetadataSize, np.nan)
        motionDirection = np.full(defaultMetadataSize, np.nan)

        # State manager
        manager = StateManager()

        #
        if returnStateValues:
            self.display.callOnFlip(manager.takeSnapshot, save=True)

        # Total frame counter
        counter = 0
        eventCounter = 0

        # Keeps track of the remaining time (in frames) until presenting a saccade-independent probe
        remainder = 0

        #
        ipresent = 0

        # Event countdown
        countdown = int(np.around(np.random.uniform(
            isirange[0],
            isirange[1],
            1
        ).item() * self.display.fps, 0))

        # Warm-up period
        for iframe in range(int(np.ceil(self.display.fps * warmup))):
            background.draw()
            timestamps[counter] = self.display.flip()
            counter += 1

        # For each trial
        for direction in self.order:

            #
            if gabor.contrast != baselineContrastLevel:
                gabor.contrast = baselineContrastLevel

            # Static period
            # self.display.state = True
            for iframe in range(int(np.ceil(self.display.fps * self.tstatic))):
                # if iframe == constants.N_SIGNAL_FRAMES:
                #     self.display.state = False
                gabor.draw()
                timestamps[counter] = self.display.flip()
                counter += 1

            # Motion onset
            # self.display.state = True
            self.display.signalEvent(3, units='frames')
            metadata[ipresent] = f'motionOnset'
            eventTimestamps[ipresent] = -1.0
            motionDirection[ipresent] = direction
            ipresent += 1
            for iframe in range(int(np.ceil(self.display.fps * self.duration))):

                #
                # sample = np.zeros(60 * 2)
                # sample[0] = 1
                # self.shared.value = int(np.random.choice(sample, 1).item())

                # Interrupt signal
                if self.shared != None and self.shared.value == -1:
                    break

                # Turn off motion onset signal
                # if iframe == constants.N_SIGNAL_FRAMES:
                #     self.display.state = False

                # Outside of probes-allowed period
                if iframe < frameIndexProbesAllowed or iframe > frameIndexProbesDisallowed:
                    gabor.phase += direction * cpf1
                    gabor.draw()
                    timestamps[counter] = self.display.flip()
                    counter += 1
                    continue

                # Inter-event interval
                if manager.inInterEventInterval:

                    # Present a probe
                    if self.shared != None and self.shared.value == 1:

                        # Progress through the state sequence
                        manager.changeState()

                        # Save the countdown (until a fictive saccade or random probe)
                        remainder = countdown

                        # Remainder must be at least the minimum ISI
                        if int(np.around(countdown / self.display.fps, 0)) < isirange[0]:
                            remainder = int(np.around(isirange[0] / self.display.fps, 0))

                        # Determine an optimal foreperiod duration
                        if foreperiodSample != None:
                            countdown = int(np.around(
                                np.random.choice(foreperiodSample, 1).item(), 0
                            ))
                        else:
                            countdown = 0

                        #
                        metadata[ipresent] = 'realtimeProbe'
                        ipresent +=1

                    # Present a fictive saccade or random probe
                    else:
                        countdown -= 1
                        if countdown == 0:

                            # Reset the remainder counter
                            remainder = 0

                            # Choose fictive saccade | random probe
                            if np.any([presentRandomProbes, presentFictiveSaccades]):

                                #
                                pathIndex = manager.choosePath(presentRandomProbes, presentFictiveSaccades)
                                manager.changeState(ipath=pathIndex)
                                print(pathIndex)

                                # Determine the next countdown
                                if pathIndex == 1:
                                    self.display.state = True
                                    countdown = int(np.around(
                                        self.display.fps * fictiveSaccadeDuration,
                                        0
                                    ))
                                    metadata[ipresent] = 'fictiveSaccade'

                                elif pathIndex == 2:
                                    self.display.state = True
                                    countdown = 0
                                    metadata[ipresent] = 'randomProbe'

                                ipresent += 1

                            # Restart countdown
                            else:
                                countdown = int(np.around(np.random.uniform(
                                    isirange[0],
                                    isirange[1],
                                    1
                                ).item() * self.display.fps, 0))

                # Foreperiod
                elif manager.inForeperiod:
                    countdown -= 1
                    if countdown <= 0:
                        manager.changeState()
                        gabor.contrast = 1
                        # self.display.state = True
                        countdown = int(np.ceil(self.display.fps * tprobe))
                        self.display.signalEvent(countdown, units='frames')

                # Perisaccadic probe presentation
                elif manager.presentingPerisaccadicProbe:
                    countdown -= 1
                    if countdown == 0:
                        manager.changeState()
                        # self.display.state = False
                        gabor.contrast = baselineContrastLevel
                        countdown = int(np.ceil(self.display.fps * timeout))
                        if self.shared.value == 1:
                            self.shared.value = 0 # Unset the shared flag

                # Refractory period
                elif manager.inRefractoryPeriod:
                    countdown -= 1
                    if countdown == 0:
                        manager.changeState()

                        # Select a new ITI
                        if remainder == 0:
                            countdown = int(np.around(np.random.uniform(
                                isirange[0],
                                isirange[1],
                                1
                            ).item() * self.display.fps, 0))

                        # Continue counting down the previous ITI
                        else:
                            countdown = remainder

                # Fictive saccade presentation
                elif manager.presentingFictiveSaccade:
                    countdown -= 1
                    if countdown == 0:
                        manager.changeState()
                        self.display.state = False
                        countdown = int(np.ceil(self.display.fps * timeout))

                # Random probe presentation
                elif manager.presentingRandomProbe:
                    countdown -= 1
                    if countdown == 0:
                        manager.changeState()
                        self.display.state = False
                        countdown = int(np.ceil(self.display.fps * timeout))

                # Increment phase
                if manager.presentingFictiveSaccade:
                    gabor.phase += direction * cpf2
                else:
                    gabor.phase += direction * cpf1

                # Draw, flip, record timestamp, and increment frame counter
                gabor.draw()
                timestamp = self.display.flip()
                timestamps[counter] = timestamp
                if manager.presentingPerisaccadicProbe:
                    if np.sum(~np.isnan(eventTimestamps)) != ipresent + 1:
                        eventTimestamps[ipresent] = timestamp
                counter += 1

            # ITI
            # self.display.state = True
            self.display.signalEvent(3, units='frames')
            metadata[ipresent] = f'motionOffset'
            eventTimestamps[ipresent] = -1.0
            motionDirection[ipresent] = direction
            ipresent += 1
            for iframe in range(int(np.ceil(self.display.fps * self.iti))):
                # if iframe == constants.N_SIGNAL_FRAMES:
                #     self.display.state = False
                background.draw()
                timestamps[counter] = self.display.flip()
                counter += 1
        #
        metadata = metadata[[len(s) > 0 for s in metadata]]
        eventTimestamps = eventTimestamps[~np.isnan(eventTimestamps)]
        motionDirection = motionDirection[~np.isnan(motionDirection)]
        self.metadata = list(zip(metadata, motionDirection, eventTimestamps))

        #
        if returnStateValues:
            return timestamps, self.metadata, np.array(manager.score)
        else:
            return timestamps, self.metadata

    def saveMetadata(self, sessionFolder):
        """
        """

        if self.metadata == None:
            return

        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            return

        with open(sessionFolderPath.joinpath('realtimeGratingMetadata.txt'), 'w') as stream:
            stream.write(f'Event, Motion, Timestamp (seconds)\n')
            for event, direction, timestamp in self.metadata:
                if timestamp == -1.0:
                    timestamp = np.nan
                stream.write(f'{event}, {direction:.0f}, {timestamp:.3f}\n')

        return

class DriftingGratingWithRandomProbe(bases.StimulusBase):
    """
    """

    def present(
        self,
        spatialFrequency=0.15,
        velocity=12,
        baselineContrast=0.3,
        probeContrastValues=(1,),
        probeContrastProbabilities=(1,),
        probeDuration=0.1,
        ipiRange=(0.5, 1),
        itiDuration=5,
        trialCount=1,
        directions=(-1, 1),
        staticPhaseDuration=3,
        motionPhaseDuration=5,
        bufferPhaseDuration=1,
        metadataArraySize=100000,
        ):
        """
        """

        self.header = {
            f'Spatial frequency': f'{spatialFrequency} (cycles/degree)',
            f'Velocity': f'{velocity} (degrees/second)',
            f'Baseline contrast': f'{baselineContrast} (0, 1)',
        }

        #
        cpp = spatialFrequency / self.display.ppd # cycles per pixel
        cpf = spatialFrequency * velocity / self.display.fps
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
            sf=cpp,
            contrast=baselineContrast,
        )

        #
        self.metadata = np.full([metadataArraySize, 4], np.nan)
        trialParameters = np.tile(directions, trialCount)
        np.random.shuffle(trialParameters)
        inIPI = False
        recordEvent = False
        eventIndex = 0
        countdown = 0

        #
        self.display.idle(itiDuration, units='seconds')

        for direction in trialParameters:

            # Static phase
            # self.display.signalEvent(3, units='frames')
            for frameIndex in range(round(self.display.fps * staticPhaseDuration)):
                gabor.draw()
                timestamp = self.display.flip()
                if frameIndex == 0:
                    self.metadata[eventIndex, :] = (1, direction, gabor.contrast, timestamp)
                    eventIndex += 1

            # Motion (buffer) phase
            # self.display.signalEvent(3, units='frames')
            for frameIndex in range(round(self.display.fps * bufferPhaseDuration)):
                gabor.phase += cpf * direction
                gabor.draw()
                timestamp = self.display.flip()
                if frameIndex == 0:
                    self.metadata[eventIndex, :] = (2, direction, gabor.contrast, timestamp)
                    eventIndex += 1

            # Motion phase
            for frameIndex in range(round(self.display.fps * motionPhaseDuration)):

                if inIPI:
                    if countdown == 0:
                        inIPI = False
                        countdown = round(self.display.fps * probeDuration)
                        self.display.signalEvent(countdown, units='frames')
                        gabor.contrast = np.random.choice(probeContrastValues, p=probeContrastProbabilities)
                        recordEvent = True

                else:
                    if countdown == 0:
                        inIPI = True
                        interval = np.random.uniform(*ipiRange, size=1).item()
                        countdown = round(self.display.fps * interval)
                        gabor.contrast = baselineContrast

                #
                countdown -= 1

                #
                gabor.phase += cpf * direction
                gabor.draw()
                timestamp = self.display.flip()
                if recordEvent:
                    self.metadata[eventIndex, :] = (3, direction, gabor.contrast, timestamp)
                    eventIndex += 1
                    recordEvent = False

            # End of motion phase: wait for the end of the IPI or wait for 1 sec 
            if inIPI:
                while countdown != 0:
                    gabor.phase += cpf * direction
                    gabor.draw()
                    self.display.flip()
                    countdown -= 1
            else:
                while countdown != 0:
                    gabor.phase += cpf * direction
                    gabor.draw()
                    self.display.flip()
                    countdown -= 1
                gabor.contrast = baselineContrast
                for frameIndex in range(round(self.display.fps * bufferPhaseDuration)):
                    gabor.phase += cpf * direction
                    gabor.draw()
                    self.display.flip()

            # ITI period
            # self.display.signalEvent(3, units='frames')
            timestamp = self.display.idle(itiDuration, units='seconds', returnFirstTimestamp=True)
            self.metadata[eventIndex, :] = (4, direction, gabor.contrast, timestamp)
            eventIndex += 1

        return
    
    def saveMetadata(self, sessionFolder):
        """
        """

        self.header.update({
            'Columns': 'Event (1=Grating, 2=Motion, 3=Probe, 4=ITI), Motion direction, Probe contrast, Timestamp'
        })
        stream = super().prepareMetadataStream(sessionFolder, 'driftingGratingMetadata', self.header)
        for array in self.metadata:
            if np.isnan(array).all():
                continue
            event, direction, contrast, timestamp = array
            line = f'{event:.0f}, {direction:.0f}, {contrast:.2f}, {timestamp:.3f}\n'
            stream.write(line)
        stream.close()

        return

class DriftingGratingWithWhiteNoise(bases.StimulusBase):
    """
    """

    def present(
        self,
        spatialFrequency=0.15,
        velocity=12,
        contrastRange=(0, 1),
        stepDuration=0.05,
        motionDirection=(-1, 1),
        trialDuration=3,
        trialCount=1,
        warmupDuration=5,
        itiDuration=5,
        staticPhaseDuration=1,
        warmupPhaseDuration=1,
        defaultMetadataSize=1000000,
        smoothContrastSequence=False,
        smoothingWindowSize=5,
        ):

        #
        self.header = {
            f'Spatial frequency': f'{spatialFrequency} (cycles/degree)',
            f'Velocity': f'{velocity} (degrees/second)',
        }
        self.metadata = np.full((defaultMetadataSize, 3), np.nan)
        # self.metadata = {
        #     'events': np.full((defaultMetadataSize, 3), np.nan),
        #     'velocity': velocity,
        #     'frequency': spatialFrequency,
        #     'interval': stepDuration
        # }

        #
        cpp = spatialFrequency / self.display.ppd # cycles per pixel
        cpf = spatialFrequency * velocity / self.display.fps
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
            sf=cpp,
        )

        #
        averageContrast = np.min(contrastRange) + np.diff(contrastRange).item() / 2

        #
        eventIndex = 0
        trials = np.repeat(motionDirection, trialCount)
        np.random.shuffle(trials)

        #
        nSteps = int(np.ceil(trialDuration / stepDuration))
        if nSteps % 2 == 0:
            nSteps += 1
        nFrames = round(self.display.fps * stepDuration)

        #
        self.display.idle(warmupDuration)
        for trialIndex, motionDirection in enumerate(trials):

            # static period - grating but no motion
            gabor.contrast = averageContrast
            for frameIndex in range(round(self.display.fps * staticPhaseDuration)):
                gabor.draw()
                self.display.flip() 

            # pre-flicker period - motion but no contrast modulation
            for frameIndex in range(round(self.display.fps * warmupPhaseDuration)):
                gabor.phase += cpf * motionDirection
                gabor.draw()
                self.display.flip()

            # Create the contrast sequence
            contrastSteps = np.random.uniform(*contrastRange, size=nSteps)
            if smoothContrastSequence:
                contrastSteps = smooth(contrastSteps, smoothingWindowSize)

            #
            for stepIndex in range(nSteps):

                #
                gabor.contrast = contrastSteps[stepIndex]
                self.display.state = not self.display.state

                #
                for frameIndex in range(nFrames):
                    gabor.draw()
                    timestamp = self.display.flip()
                    if frameIndex == 0:
                        self.metadata[eventIndex, :] = (gabor.contrast, motionDirection, timestamp)
                        eventIndex += 1
                    gabor.phase += cpf * motionDirection

            # Return the display patch to a LOW state
            if self.display.state:
                self.display.state = False
                self.display.drawBackground()
                timestamp = self.display.flip()
                self.metadata[eventIndex, :] = (np.nan, np.nan, timestamp)
                eventIndex += 1

            # ITI period
            self.display.idle(itiDuration)

        #
        self.metadata = np.array([
            entry for entry in self.metadata
                if np.isnan(entry).all() == False
        ])

        return
    
    def saveMetadata(self, sessionFolder):
        """
        """

        self.header.update({
            'Columns': 'Contrast (0, 1), Motion direction (-1=Left, 1=Right), Timestamp (seconds)'
        })
        stream = super().prepareMetadataStream(sessionFolder, 'noisyGratingMetadata', self.header)
        for c, m, t in self.metadata:
            line = f'{c:.3f}, {m:.0f}, {t:.3f}\n'
            stream.write(line)

        return

class DriftingGratingWithWhiteNoise2(bases.StimulusBase):
    """
    """

    def present(
        self,
        spatialFrequency=0.15,
        velocity=12,
        contrastRange=(0.4, 1),
        stepDuration=0.033,
        motionDirection=(-1, 1),
        trialDuration=3,
        trialCount=1,
        warmupDuration=5,
        itiDuration=5,
        staticPhaseDuration=3,
        defaultMetadataSize=1000000,
        ):

        #
        self.header = {
            f'Spatial frequency': f'{spatialFrequency} (cycles/degree)',
            f'Velocity': f'{velocity} (degrees/second)',
        }
        self.metadata = np.full((defaultMetadataSize, 3), np.nan)

        #
        cpp = spatialFrequency / self.display.ppd # cycles per pixel
        cpf = spatialFrequency * velocity / self.display.fps
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
            sf=cpp,
        )

        #
        eventIndex = 0
        trials = np.repeat(motionDirection, trialCount)
        np.random.shuffle(trials)

        #
        nSteps = int(np.ceil(trialDuration / stepDuration))
        nFramesPerStep = round(self.display.fps * stepDuration)
        if nFramesPerStep <= 1:
            print(f'Warning: Increasing step duration to at least 2 frames')
            nFramesPerStep = 2

        #
        self.display.idle(warmupDuration)
        for trialIndex, motionDirection in enumerate(trials):

            # static period/no motion
            gabor.contrast = np.min(contrastRange) + np.diff(contrastRange).item() / 2
            for frameIndex in range(round(self.display.fps * staticPhaseDuration)):
                gabor.draw()
                self.display.flip() 

            #
            for stepIndex in range(nSteps):

                #
                gabor.contrast = np.random.uniform(*contrastRange, size=1).item()
                # self.display.signalEvent(1, units='frames')
                self.display.state = True

                #
                for frameIndex in range(nFramesPerStep):
                    if frameIndex == 2:
                        self.display.state = False
                    gabor.draw()
                    timestamp = self.display.flip()
                    if frameIndex == 0:
                        self.metadata[eventIndex, :] = (gabor.contrast, motionDirection, timestamp)
                        eventIndex += 1
                    gabor.phase += cpf * motionDirection

            # Return the display patch to a LOW state
            # self.display.state = False

            # ITI period
            self.display.idle(itiDuration)

        #
        self.metadata = np.array([
            entry for entry in self.metadata
                if np.isnan(entry).all() == False
        ])

        return
    
    def saveMetadata(self, sessionFolder):
        """
        """

        self.header.update({
            'Columns': 'Contrast (0, 1), Motion direction (-1, 1), Timestamp (seconds)'
        })
        stream = super().prepareMetadataStream(sessionFolder, 'noisyGratingMetadata', self.header)
        for c, m, t in self.metadata:
            line = f'{c:.3f}, {m:.0f}, {t:.3f}\n'
            stream.write(line)

        return
