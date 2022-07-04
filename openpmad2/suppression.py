import numpy as np
import pathlib as pl
import multiprocessing as mp
from psychopy import visual
from openpmad2 import warping
from openpmad2 import writing
from openpmad2 import testing
from openpmad2 import helpers
from openpmad2 import constants
from openpmad2.displays import WarpedWindow

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
        tdelay=None,
        warmup=1,
        timeout=3,
        tprobe=0.05,
        tmargin=3,
        isirange=(1, 3),
        presentRandomProbes=True,
        presentFictiveSaccades=True,
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
        eventTimestamps = np.full(5000, np.nan)

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
                        self.display.flashSignalPatch(frameCount=countdown)

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
            for iframe in range(int(np.ceil(self.display.fps * self.iti))):
                # if iframe == constants.N_SIGNAL_FRAMES:
                #     self.display.state = False
                background.draw()
                timestamps[counter] = self.display.flip()
                counter += 1
        #
        metadata = metadata[[len(s) > 0 for s in metadata]]
        eventTimestamps = eventTimestamps[~np.isnan(eventTimestamps)]
        self.metadata = list(zip(metadata, eventTimestamps))

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
            stream.write(f'Event, Timestamp (seconds)\n')
            for event, timestamp in self.metadata:
                stream.write(f'{event}, {timestamp:.3f}\n')

        return
