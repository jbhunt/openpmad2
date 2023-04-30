import numpy as np
import pathlib as pl
from psychopy import visual
from psychopy import core
import serial

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
            self.display.signalEvent(3, units='frames')
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

class StaticGratingWithVariableProbe():
    """
    """

    def __init__(self, display):
        self.display = display
        self.metadata = None
        return

    def present(
        self,
        spatialFrequency=0.08,
        probeDuration=0.5,
        probeContrastLevels=(0.01, 0.02, 0.03, 0.05, 0.1),
        probeContrastProbabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
        baselineContrastLevel=0.5,
        interProbeIntervalRange=(5, 10),
        trialCount=5,
        ):

        #
        if len(probeContrastLevels) != len(probeContrastProbabilities):
            raise Exception('Number of probe levels must equal the number of probabilities')

        if np.sum(probeContrastProbabilities) != 1:
            raise Exception('Probe probabilities must sum to 1')

        #
        clock = core.MonotonicClock()
        self.metadata = np.full([trialCount, 3], np.nan)
        self.metadata[:, 0] = np.arange(1, trialCount + 1, 1)
        self.metadata[:, 1] = np.random.choice(probeContrastLevels, trialCount, p=probeContrastProbabilities)
        self.metadata[:, 1]

        cpp = spatialFrequency / self.display.ppd # cycles per pixel
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
            sf=cpp,
            contrast=baselineContrastLevel,
        )

        #
        gabor.draw()
        self.display.flip()
        for trialIndex, probeContrastLevel in enumerate(self.metadata[:, 1]):
            interProbeInterval = np.random.randint(
                round(interProbeIntervalRange[0] * self.display.fps),
                round(interProbeIntervalRange[1] * self.display.fps),
                1
            ).item()
            for frameIndex in range(interProbeInterval):
                gabor.draw()
                self.display.flip()
            gabor.contrast = probeContrastLevel
            self.display.signalEvent(3, units='frames')
            self.metadata[trialIndex, -1] = clock.getTime()
            for frameIndex in range(round(probeDuration * self.display.fps)):
                gabor.draw()
                self.display.flip()
            gabor.contrast = baselineContrastLevel

        #
        for frameIndex in range(round(3 * self.display.fps)):
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
            stream.write(f'Trial number, Probe contrast, Timestamp\n')
            for trialNumber, probeContrast, timestamp in self.metadata:
                stream.write(f'{trialNumber:.0f}, {probeContrast:.3f}, {timestamp:.3f}\n')

        return

class DriftingGratingWithVariableProbe():
    """
    """

    def __init__(self, display):
        """
        """

        self.display = display
        self.metadata = None

        return

    def present(
        self,
        spatialFrequency=0.08,
        velocity=12,
        probeDuration=0.5,
        blockDuration=5,
        nBlocksPerDirection=1,
        probeContrastLevels=(1, 1),
        probeContrastProbabilities=(0.5, 0.5),
        baselineContrastLevel=0.5,
        interProbeIntervalRange=(1, 3),
        interBlockInterval=5,
        buffers=(0.5, 0.5),
        ):
        """
        """

        #
        if len(probeContrastLevels) != len(probeContrastProbabilities):
            raise Exception('Number of probe levels must equal the number of probabilities')

        if np.sum(probeContrastProbabilities) != 1:
            raise Exception('Probe probabilities must sum to 1')

        #
        self.metadata = np.full([1000000, 4], np.nan)

        #
        cpp = spatialFrequency / self.display.ppd # cycles per pixel
        cpf = spatialFrequency * velocity / self.display.fps
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
            sf=cpp,
            contrast=baselineContrastLevel,
        )

        #
        directions = np.tile([-1, 1], nBlocksPerDirection)
        np.random.shuffle(directions)
        countdown = int(np.ceil(np.random.uniform(*interProbeIntervalRange, size=1).item()))
        trialIndex = 0
        presentingProbe = False
        recordTimestamp = False

        for direction in directions:

            # Show grating static for 3 seconds
            for iFrame in range(int(np.ceil(3 * self.display.fps))):
                gabor.draw()
                self.display.flip()

            #
            nFrames = int(np.ceil(blockDuration * self.display.fps))
            boundaries = (
                0 + int(np.ceil(buffers[0] * self.display.fps)),
                nFrames - int(np.ceil(buffers[1] * self.display.fps))
            )
            probeDurationInFrames = int(np.ceil(probeDuration * self.display.fps))
            for iFrame in range(nFrames):

                #
                if countdown == 0:

                    # Exit probe phase
                    if presentingProbe:

                        # Choose an new inter-probe interval
                        countdown = int(np.ceil(np.random.uniform(*interProbeIntervalRange, size=1).item() * self.display.fps))
                        gabor.contrast = baselineContrastLevel
                        presentingProbe = False

                    # Attempt to enter probe phase
                    else:

                        # Make sure the probe onset and offset are within the temporal boundaries
                        if iFrame < boundaries[0] or iFrame + probeDurationInFrames > boundaries[1]:
                            countdown = 1 # Keep reseting the countdown to 1

                        # Present the probe
                        else:
                            self.display.signalEvent(3, units='frames')
                            gabor.contrast = np.random.choice(probeContrastLevels, p=probeContrastProbabilities)
                            self.metadata[trialIndex, 0] = trialIndex + 1
                            self.metadata[trialIndex, 1] = gabor.contrast
                            self.metadata[trialIndex, 2] = direction
                            countdown = int(np.ceil(probeDuration * self.display.fps))
                            presentingProbe = True
                            recordTimestamp = True
    
                #
                gabor.draw()
                timestamp = self.display.flip()
                if recordTimestamp:
                    self.metadata[trialIndex, 3] = timestamp
                    trialIndex += 1 # Increment the trial index
                    recordTimestamp = False
                gabor.phase += direction * cpf
                countdown -= 1
            
            # Inter-block interval
            self.display.idle(interBlockInterval, units='seconds')

        #
        mask = np.invert(np.isnan(self.metadata).all(axis=1))
        self.metadata = self.metadata[mask, :]

        return

    def saveMetadata(self, sessionFolder):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)

        if self.metadata is None:
            return

        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        filename = str(sessionFolderPath.joinpath('driftingGratingWithProbeMetadata.txt'))
        with open(filename, 'w') as stream:
            stream.write(f'Trial number, Probe contrast, Motion, Timestamp\n')
            for trialNumber, probeContrast, motion, timestamp in self.metadata:
                stream.write(f'{trialNumber:.0f}, {probeContrast:.3f}, {motion:.0f}, {timestamp:.3f}\n')

        return
    
class StaticGratingWithRealtimeProbe():
    """
    """

    def __init__(self):
        """
        """

        self._connection = None

        return
    
    def _connectWithMicrocontroller(
        self,
        baudrate=9600,
        timeout=1,
        ):
        """
        """

        #
        if self._connection is not None:
            self._connection.close()
            self._connection = None

        outgoing = bytes('a', 'utf-8')
        connected = False
        devices = pl.Path('/dev/').glob('*ttyACM*')
        for device in devices:
            try:
                connection = serial.Serial(
                    str(device),
                    baudrate,
                    timeout=timeout
                )
            except Exception as error:
                continue
            connection.write(outgoing)
            incoming = connection.read(1)
            if len(incoming) > 0 and incoming == outgoing:
                connected = True
                self._connection = connection
                break

        #
        if connected == False:
            raise Exception('Failed to connect with microcontroller')

        return
    
    def present(
        self,
        spatialFrequency=0.08,
        baselineContrastLevel=0.5,
        probeContrastLevels=(1,),
        probeContrastProbabilities=(1.0,),
        probeDuration=0.05,
        sessionLength=5,
        ):
        """
        """

        self._connectWithMicrocontroller()

        #
        cpp = spatialFrequency / self.display.ppd # cycles per pixel
        gabor = visual.GratingStim(
            self.display,
            size=self.display.size,
            units='pix',
            sf=cpp,
            contrast=baselineContrastLevel,
        )

        #
        nTrials = round(sessionLength * self.display.fps / 2, 0)
        self.metadata = np.full([nTrials, 2], np.nan)

        #
        iTrial = 0
        nFrames = round(self.display.fps * sessionLength, 0)
        countdown = 0
        recordTimestamp = False
        for iFrame in range(nFrames):

            #
            if countdown == 0 and gabor.contrast != baselineContrastLevel:
                gabor.contrast = baselineContrastLevel

            #
            if self._connection.inWaiting > 0:
                message = self._connection.read()
                gabor.contrast = np.random.choice(probeContrastLevels, p=probeContrastProbabilities, size=1).item()
                countdown = round(self.display.fps * probeDuration)
                self.metadata[iTrial, 0] = gabor.contrast
                recordTimestamp = True

            #
            gabor.draw()
            timestamp = self.display.flip()
            if recordTimestamp: 
                self.metadata[iTrial, 1] = timestamp
                iTrial += 1
            
            #
            if countdown != 0:
                countdown -= 1

        return
    
    def saveMetadata(self, sessionFolder):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)

        if self.metadata is None:
            return

        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        filename = str(sessionFolderPath.joinpath('staticGratingWithRealtimeProbeMetadata.txt'))
        with open(filename, 'w') as stream:
            stream.write(f'Probe contrast, Timestamp\n')
            for probeContrast, timestamp in self.metadata:
                stream.write(f'{probeContrast:.3f}, {timestamp:.3f}\n')
