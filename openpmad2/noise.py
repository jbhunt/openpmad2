from . import bases
from . import warping
from . import writing
from . import helpers
from . import constants
from psychopy import visual
import pickle
import numpy as np
import pathlib as pl
from PIL import Image
from openpmad2.constants import numpyRandomSeed

#
np.random.seed(numpyRandomSeed)

def _computeGridPoints(lengthInDegrees, display):
    """
    Compute the coordinates for each square in a grid which uniformly
    covers the entire display

    keywords
    --------
    lengthInDegrees
        The side length of a unit grid square (in degrees of visual angle)
    """

    lengthInPixels = lengthInDegrees * display.ppd

    #
    shape = np.array([
        int(display.width  // lengthInPixels),
        int(display.height // lengthInPixels)
    ])
    offset = np.array([
        round(display.width  % lengthInPixels / 2, 2),
        round(display.height % lengthInPixels / 2, 2)
    ])

    #
    xi, yi = np.meshgrid(
        np.arange(shape[0]) * lengthInPixels + (lengthInPixels / 2),
        np.arange(shape[1]) * lengthInPixels + (lengthInPixels / 2)
    )
    xi = xi + offset[0] - (display.width / 2)
    yi = yi + offset[1] - (display.height / 2)

    #
    coordinates = np.hstack([
        xi.reshape(-1, 1),
        yi.reshape(-1, 1)
    ])

    return np.around(coordinates, 0), np.flip(shape)

class SparseNoise(bases.StimulusBase):
    """
    """

    def _generateMetadata(
        self,
        coordsInPixels,
        repeats,
        field,
        nTrials,
        gridShape,
        randomize,
        correctVerticalReflection
        ):
        """
        """

        #
        coordsInDegrees = np.around(coordsInPixels / self.display.ppd, 3)
        gridHeight, gridWidth = gridShape
        self.metadata = {

            # Luminance values for each subregion on the ith trial
            'fields': np.full([nTrials, gridHeight, gridWidth], np.nan),

            # x and y coordinates in degrees for the center of the illuminated subregion for the ith trial
            'coords': np.full([nTrials, 2], np.nan),

            # indices which indicates the subregion illuminated on the ith trial
            'indices': np.full([nTrials, 1], 0).astype(np.int64),
                                                       
            # A list of event names (either spot or flash)
            'events': np.full([nTrials * 2, 1], '').astype(np.chararray)
        }

        #
        iTrial = 0
        for iRepeat in range(repeats):
            for iSubregion in np.arange(field.nElements):
                image = np.full(field.nElements, -1)
                image[iSubregion] = 1
                self.metadata['fields'][iTrial] = image.reshape(gridShape)
                coords = coordsInDegrees[iSubregion]
                if correctVerticalReflection:
                    coords[1] = coords[1] * -1
                self.metadata['coords'][iTrial] = coords
                self.metadata['indices'][iTrial] = iSubregion
                iTrial += 1

        #
        if randomize:
            trialIndices = np.arange(nTrials)
            np.random.shuffle(trialIndices)
            self.metadata['fields'] = self.metadata['fields'][trialIndices]
            self.metadata['coords'] = self.metadata['coords'][trialIndices]
            self.metadata['indices'] = self.metadata['indices'][trialIndices]

        return

    def _runMainLoop(
        self,
        field,
        tIdle,
        cycle,
        nTrials,
        nTrialsBetweenSignals,
        ):
        """
        """

        if self.display.backgroundColor != -1:
            self.display.backgroundColor = -1
        self.display.idle(tIdle)

        iEvent = 0
        for iTrial in range(nTrials):

            # Only signal the trial once every N trials
            # This can be disabled by setting 'nTiralsBetweenSignals' equal to 1
            if iTrial % nTrialsBetweenSignals == 0:
                signal = True
            else:
                signal = False

            #
            iSubregion = self.metadata['indices'][iTrial]
            colors = np.full(field.nElements, -1)
            colors[iSubregion] = 1
            field.colors = colors.reshape(-1, 1)
            
            #
            if signal:
                self.display.signalEvent(3, units='frames')
                self.metadata['events'][iEvent] = 'spot onset'
                iEvent += 1
            for iFrame in range(int(np.ceil(cycle[0] * self.display.fps))):
                field.draw()
                self.display.flip()

            #
            colors = np.full(field.nElements, -1).reshape(-1, 1)
            field.colors = colors

            #
            if signal:
                self.display.signalEvent(3, units='frames')
                self.metadata['events'][iEvent] = 'spot offset'    
                iEvent += 1
            for iFrame in range(int(np.ceil(cycle[1] * self.display.fps))):
                field.draw()
                self.display.flip()

        #
        self.display.idle(tIdle)

        return

    def present(
        self,
        radius=5,
        cycle=(0.5, 0.5),
        repeats=1,
        tIdle=3,
        randomize=True,
        correctVerticalReflection=True,
        nTrialsBetweenSignals=1,
        ):
        """
        """

        #
        length = radius * 2
        coordsInPixels, gridShape = _computeGridPoints(length, self.display)
        gridHeight, gridWidth = gridShape
        nSubregions = coordsInPixels.shape[0]

        #
        nTrials = int(gridWidth * gridHeight * repeats)

        #
        field = visual.ElementArrayStim(
            self.display,
            fieldPos=coordsInPixels,
            fieldShape='sqr',
            nElements=nSubregions,
            sizes=length * self.display.ppd,
            colors=np.random.choice([-1, 1], size=nSubregions).reshape(-1, 1),
            elementMask='circle',
            elementTex=None,
            units='pixels', 
        )

        # Generate metadata
        self._generateMetadata(
            coordsInPixels,
            repeats,
            field,
            nTrials,
            gridShape,
            randomize,
            correctVerticalReflection
        )
        self.metadata['length'] = length
        self.metadata['cycle'] = cycle

        # Run main loop
        self._runMainLoop(
            field,
            tIdle,
            cycle,
            nTrials,
            nTrialsBetweenSignals
        )

        #
        mask = np.array([len(line.item()) > 0 for line in self.metadata['events']])
        self.metadata['events'] = self.metadata['events'][mask].flatten()

        return

    def saveMetadata(self, sessionFolder):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        #
        with open(sessionFolderPath.joinpath('sparseNoiseMetadata.pkl'), 'wb') as stream:
            pickle.dump(self.metadata, stream)

        return    

class SimpleBinaryNoise(bases.StimulusBase):
    """
    """
    
    def _generateMetadata(
        self,
        gridShape,
        coordsInDegrees,
        correctVerticalReflection,
        length,
        tImage,
        nImagesBetweenFlashes,
        nImages,
        pHigh,
        ):
        """
        """

        # Take care of the easy stuff
        gridHeight, gridWidth = gridShape
        nSubregions = gridHeight * gridWidth

        #
        self.metadata = {
            'events': list(),
            'fields': list(),
            'coords': coordsInDegrees,
            'length': length,
            'interval': tImage
        }

        # Reflect coordinates across the horizontal axis
        if correctVerticalReflection:
            self.metadata['coords'][:, 1] = self.metadata['coords'][:, 1] * -1

        #
        countdown = nImagesBetweenFlashes

        for iField in range(nImages):

            # Full-field flash
            if countdown == 0:
                
                #
                for event in ('flash onset', 'flash offset'):
                    self.metadata['events'].append(event)
                    self.metadata['fields'].append(np.full(gridShape, np.nan))
                countdown = nImagesBetweenFlashes

            # Generate new field
            colors = np.random.choice(
            a=np.array([-1, 1]),
            p=np.array([1 - pHigh, pHigh]),
            size=nSubregions
            ).reshape(*gridShape)
            self.metadata['fields'].append(colors)
            self.metadata['events'].append('field onset')
            countdown -= 1

        # Cast to numpy arrays
        self.metadata['fields'] = np.array(self.metadata['fields'])
        self.metadata['events'] = np.array(self.metadata['events'], dtype=object).reshape(-1, 1)

        return
    
    def _runMainLoop(
        self,
        tImage,
        tIdle,
        field,
        nTrialsBetweenSignals,
        ):
        """
        """

        # Change the background to black and wait 5 seconds
        if self.display.backgroundColor != -1:
            self.display.backgroundColor = -1
        self.display.idle(tIdle, units='seconds')

        #
        nTrials = self.metadata['fields'].shape[0]
        iterable = zip(
            range(nTrials),
            self.metadata['events'],
            self.metadata['fields']
        )
        for iTrial, event, colors in iterable:

            # Only signal the trial once every N trials
            # This can be disabled by setting 'nTiralsBetweenSignals' equal to 1
            if iTrial % nTrialsBetweenSignals == 0:
                signal = True
            else:
                signal = False

            # Full-field flash onset
            if event == 'flash onset':
                self.display.setBackgroundColor(1)
                if signal:
                    self.display.signalEvent(3, units='frames')
                for iFrame in range(int(np.ceil(tImage * self.display.fps))):
                    self.display.drawBackground()
                    self.display.flip()

            # Full-field flash offset
            elif event == 'flash offset':
                self.display.setBackgroundColor(-1)
                if signal:
                    self.display.signalEvent(3, units='frames')
                for iFrame in range(int(np.ceil(tImage * self.display.fps))):
                    self.display.drawBackground()
                    self.display.flip()

            # Present the stimulus field
            elif event == 'field onset':

                #
                field.colors = colors.reshape(-1, 1)

                #
                if signal:
                    self.display.signalEvent(3, units='frames')

                #
                for iFrame in range(int(np.ceil(tImage * self.display.fps))):
                    field.draw()
                    self.display.flip()

        # Display black screen for 5 seconds
        self.display.idle(tIdle, units='seconds')

        return

    def present(
        self,
        length=5,
        tImage=0.5,
        nImages=10,
        pHigh=0.2,
        tIdle=3,
        correctVerticalReflection=True,
        nImagesBetweenFlashes=5,
        nTrialsBetweenSignals=1,
        ):
        """
        """

        coordsInPixels, (gridHeight, gridWidth) = _computeGridPoints(length, self.display)
        coordsInDegrees = coordsInPixels / self.display.ppd
        nSubregions = coordsInPixels.shape[0]
        initialColors = np.random.choice([-1, 1], p=[1 - pHigh, pHigh], size=nSubregions).reshape(-1, 1)

        # Create the visual field
        field = visual.ElementArrayStim(
            self.display,
            fieldPos=coordsInPixels,
            fieldShape='sqr',
            nElements=nSubregions,
            sizes=length * self.display.ppd,
            colors=initialColors,
            elementMask=None,
            elementTex=None,
            units='pixels', 
        )

        # Populate the metadata dictionary
        gridShape = (gridHeight, gridWidth)
        self._generateMetadata(
            gridShape,
            coordsInDegrees,
            correctVerticalReflection,
            length,
            tImage,
            nImagesBetweenFlashes,
            nImages,
            pHigh,
        )

        #
        self._runMainLoop(
            tImage,
            tIdle,
            field,
            nTrialsBetweenSignals,
        )

        # Clean up the events metadata array
        mask = np.array([len(line.item()) > 0 for line in self.metadata['events']])
        self.metadata['events'] = self.metadata['events'][mask].flatten()

        return

    def saveMetadata(self, sessionFolder):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        #
        with open(sessionFolderPath.joinpath('binaryNoiseMetadata.pkl'), 'wb') as stream:
            pickle.dump(self.metadata, stream)

        return

class JitteredBinaryNoise(bases.StimulusBase):
    """
    """

    def _randomizeTrials(
        self,
        ):
        """
        """

        nTrials = len(
            self.metadata['events'])
        shuffledTrialIndices = np.random.choice(
            np.arange(nTrials),
            size=nTrials
        )

        for key, oldList in self.metadata.items():

            #
            newList = list()
            for iTrial in shuffledTrialIndices:
                newList.append(oldList[iTrial])

            #
            self.metadata[key] = newList

        return

    def _insertFlashes(
        self,
        cycle,
        nSubregions,
        nTrialsBetweenFlashes
        ):
        """
        """

        # Insert full-field flashes
        for key, oldList in self.metadata.items():

            # Determine the fill values
            if key == 'events':
                inserts = ('flash onset', 'flash offset')
                dtype = object
            elif key == 'shifted':
                inserts = (False, False)
                dtype = bool
            elif key == 'fields':
                shape = (nSubregions, 1)
                inserts = (
                    np.full(shape, True),
                    np.full(shape, False),
                )
                dtype = bool
            elif key == 'colors':
                shape = (nSubregions, 1)
                inserts = (
                    np.full(shape, 1),
                    np.full(shape, -1),
                )
                dtype = float
            elif key == 'offsets':
                inserts = (
                    np.full(2, np.nan),
                    np.full(2, np.nan)
                )
                dtype = float

            # Create the new list
            newList = list()
            if cycle[1] != 0:
                factor = 2
            else:
                factor = 1
            countdown = nTrialsBetweenFlashes * factor
            for iElement, element in enumerate(oldList):
                if countdown == 0:
                    for insert in inserts:
                        newList.append(insert)
                    countdown = nTrialsBetweenFlashes * factor
                newList.append(element)
                countdown -= 1

            #
            self.metadata[key] = np.array(newList, dtype=dtype)

        return

    def _generateMetadata(
        self,
        nImages,
        repeats,
        pHigh,
        nSubregions,
        shiftInDegrees,
        coordsInPixels,
        randomize,
        length,
        correctVerticalReflection,
        nTrialsBetweenFlashes,
        cycle,
        gridShape
        ):
        """
        """

        self.metadata = {
            'events': list(),
            'fields': list(),
            'colors': list(),
            'offsets': list(),
            'shifted' : list()
        }

        #
        for iTrial in range(nImages):

            # Generate a new field
            mask = np.random.choice(
                np.array([True, False]),
                p=np.array([pHigh, 1 - pHigh]),
                size=nSubregions
            ).reshape(-1, 1)

            # Create a colors array from the field
            colors = np.array([
                1 if flag else -1
                    for flag in mask.flatten()
            ]).reshape(-1, 1)

            # Shift (or don't shift) the field
            for phase in ('original', 'shifted'):
                
                # Choose an offset
                if phase == 'shifted':
                    offset = np.array([shiftInDegrees, shiftInDegrees]) * np.random.choice([-1, 1], size=2)
                elif phase == 'original':
                    offset = np.array([0, 0])

                #
                for iRepeat in range(repeats):
                    for event in ('field onset', 'field offset'):

                        #
                        if event == 'field onset':
                            self.metadata['fields'].append(mask)
                            self.metadata['colors'].append(colors)
                            self.metadata['offsets'].append(offset)
                            self.metadata['shifted'].append(True if phase == 'shifted' else False)
                            self.metadata['events'].append(event)

                        #
                        elif event == 'field offset' and cycle[-1] != 0:
                            self.metadata['fields'].append(np.full([nSubregions, 1], False))
                            self.metadata['colors'].append(np.full([nSubregions, 1], np.nan))
                            self.metadata['offsets'].append(np.full(2, np.nan))
                            self.metadata['shifted'].append(False)
                            self.metadata['events'].append(event)                       

        # Randomize trials
        if randomize:
            self._randomizeTrials()

        #
        if nTrialsBetweenFlashes is not None:
            self._insertFlashes(
                cycle,
                nSubregions,
                nTrialsBetweenFlashes
            )

        #
        self.metadata['length'] = length
        self.metadata['coords'] = coordsInPixels
        self.metadata['shape'] = gridShape

        #
        if correctVerticalReflection:
            self.metadata['coords'][:, 1] = self.metadata['coords'][:, 1] * -1

        return
    
    def _runMainLoop(
        self,
        field,
        cycle,
        tIdle,
        nTrialsBetweenSignals,
        ):
        """
        """

        # Change the background to black and idle
        if self.display.backgroundColor != -1:
            self.display.backgroundColor = -1
        self.display.idle(tIdle, units='seconds')

        #
        originFieldPosition = field.fieldPos
        nFramesOnPhase = int(np.ceil(self.display.fps * cycle[0]))
        nFramesOffPhase = int(np.ceil(self.display.fps * cycle[1]))

        #
        iterable = zip(
            self.metadata['events'],
            self.metadata['colors'],
            self.metadata['offsets']
        )
        for iTrial, (event, colors, offset) in enumerate(iterable):

            # Only signal the trial once every N trials
            # This can be disabled by setting 'nTiralsBetweenSignals' equal to 1
            if iTrial % nTrialsBetweenSignals == 0:
                signal = True
            else:
                signal = False

            #
            if event == 'field onset':
                self.display.clearBuffer()
                field.fieldPos = offset * self.display.ppd + originFieldPosition
                field.colors = colors
                methodToCall = field.draw
                nFramesToDraw = nFramesOnPhase

            #
            elif event == 'field offset':
                self.display.setBackgroundColor(-1)
                methodToCall = self.display.drawBackground
                nFramesToDraw = nFramesOffPhase

            #
            elif event == 'flash onset':
                self.display.setBackgroundColor(1)
                methodToCall = self.display.drawBackground
                nFramesToDraw = nFramesOnPhase

            #
            elif event == 'flash offset':
                self.display.setBackgroundColor(-1)
                methodToCall = self.display.drawBackground
                nFramesToDraw = nFramesOffPhase

            if signal:
                self.display.signalEvent(3, units='frames')
            for iFrame in range(nFramesToDraw):
                methodToCall()
                self.display.flip()

        #
        self.display.idle(tIdle)

        return

    def present(
        self,
        length=10,
        repeats=1,
        nImages=20,
        cycle=(0.5, 0.5),
        pHigh=0.2,
        randomize=True,
        tIdle=3,
        correctVerticalReflection=True,
        nTrialsBetweenFlashes=5,
        nTrialsBetweenSignals=1,
        ):
        """
        """

        #
        coordsInPixels, gridShape = _computeGridPoints(length, self.display)
        nSubregions = coordsInPixels.shape[0]
        initialColors = np.random.choice([-1, 1], p=[1 - pHigh, pHigh], size=nSubregions).reshape(-1, 1)

        # Create the visual field
        field = visual.ElementArrayStim(
            self.display,
            fieldPos=coordsInPixels,
            fieldShape='sqr',
            nElements=nSubregions,
            sizes=length * self.display.ppd,
            colors=initialColors,
            elementMask=None,
            elementTex=None,
            units='pixels', 
        )

        # Create the metadata container
        # shiftInPixels = round(length / 2 * self.display.ppd, 2)
        shiftInDegrees = round(length / 2, 2)
        self._generateMetadata(
            nImages,
            repeats,
            pHigh,
            nSubregions,
            shiftInDegrees,
            coordsInPixels,
            randomize,
            length,
            correctVerticalReflection,
            nTrialsBetweenFlashes,
            cycle,
            gridShape
        )


        # Run main presentation loop
        self._runMainLoop(
            field,
            cycle,
            tIdle,
            nTrialsBetweenSignals
        )

        return

    def saveMetadata(self, sessionFolder):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        #
        with open(sessionFolderPath.joinpath('jitteredNoiseMetadata.pkl'), 'wb') as stream:
            pickle.dump(self.metadata, stream)

        return
    
class JitteredBinaryNoise2(bases.StimulusBase):
    """
    """

    def _generateMetadata(
        self,
        nImagesPerBlock,
        nBlocksPerCondition,
        nBlockRepeats,
        includeJitteredBlocks,
        pSubregionHigh,
        nSubregions,
        ):
        """
        """

        #
        nConditions = 2 if includeJitteredBlocks else 1
        nUniqueImages = nConditions * nBlocksPerCondition * nBlockRepeats * nImagesPerBlock

        #
        self.metadata = {
            'blocks': np.full([nUniqueImages, 1], np.nan),
            'events': np.full([10000000, 1], '').astype(object),
            'values': np.full([nUniqueImages, nSubregions, 1], np.nan),
            'jittered' : np.full([nUniqueImages, 1], False)
        }

        #
        for iTrial in range(nUniqueImages):
            values = np.random.choice(
                np.array([1, -1]),
                p=np.array([pSubregionHigh, 1 - pSubregionHigh]),
                size=nSubregions
            ).reshape(-1, 1)
            self.metadata['values'][iTrial] = values

        return
    
    def _runMainLoop(
        self,
        field,
        fieldCycle,
        nImagesPerBlock,
        nBlocksPerCondition,
        nBlockRepeats,
        includeJitteredBlocks,
        nSubregions,
        offsetInPixels,
        tIdle,
        nTrialsBetweenFlashes,
        flashCycle,
        nSignalFramesForField,
        nSignalFramesForFlash
        ):
        """
        """

        #
        if includeJitteredBlocks:
            conditions = ('original', 'jittered')
        else:
            conditions = ('original')

        #
        gridNodes = field.fieldPos

        #
        if self.display.backgroundColor != -1:
            self.display.setBackgroundColor(-1)
        for iFrame in range(int(np.ceil(self.display.fps * tIdle))):
            self.display.drawBackground()
            self.display.flip()

        #
        iTrial = 0
        iEvent = 0
        iBlock = 0
        for condition in conditions:
            for iBlock_ in range(nBlocksPerCondition):
                for iRepeat in range(nBlockRepeats):
                    for iField in range(nImagesPerBlock):

                        # Present the full-field flash
                        if iTrial % nTrialsBetweenFlashes == 0:

                            #
                            self.metadata['events'][iEvent] = 'flash onset'
                            iEvent += 1
                            self.display.signalEvent(nSignalFramesForFlash, units='frames')
                            self.display.setBackgroundColor(1)
                            for iFrame in range(int(np.ceil(self.display.fps * flashCycle[0]))):
                                self.display.drawBackground()
                                self.display.flip()

                            #
                            self.metadata['events'][iEvent] = 'flash offset'
                            iEvent += 1
                            self.display.setBackgroundColor(-1)
                            self.display.signalEvent(nSignalFramesForFlash, units='frames')
                            for iFrame in range(int(np.ceil(self.display.fps * flashCycle[1]))):
                                self.display.drawBackground()
                                self.display.flip()

                        #
                        values = self.metadata['values'][iTrial].reshape(nSubregions, 1)
                        field.colors = values
                        if condition == 'jittered':
                            field.fieldPos = gridNodes + offsetInPixels
                            self.metadata['jittered'][iTrial] = True
                        else:
                            self.metadata['jittered'][iTrial] = False

                        #
                        self.metadata['events'][iEvent] = 'field onset'
                        iEvent += 1
                        self.display.signalEvent(nSignalFramesForField, units='frames')
                        for iFrame in range(int(np.ceil(self.display.fps * fieldCycle[0]))):
                            field.draw()
                            timestamp = self.display.flip()

                        #
                        self.metadata['events'][iEvent] = 'field offset'
                        iEvent += 1
                        self.display.signalEvent(nSignalFramesForField, units='frames')
                        for iFrame in range(int(np.ceil(self.display.fps * fieldCycle[1]))):
                            self.display.drawBackground()
                            self.display.flip()

                        #
                        self.metadata['blocks'][iTrial] = iBlock + 1

                        #
                        iTrial += 1

                    #
                    iBlock += 1

        #
        mask = np.array([
            len(element.item()) > 0
                for element in self.metadata['events']
        ])
        self.metadata['events'] = self.metadata['events'][mask, :]

        return
    

    def present(
        self,
        length=10,
        fieldCycle=(0.5, 0.5),
        nBlockRepeats=1,
        nImagesPerBlock=10,
        nBlocksPerCondition=1,
        includeJitteredBlocks=True,
        jitterDirection=(1, 1),
        pSubregionHigh=0.2,
        tIdle=3,
        correctVerticalReflection=True,
        nTrialsBetweenFlashes=5,
        flashCycle=(1, 1),
        nSignalFramesForField=3,
        nSignalFramesForFlash=6,
        ):
        """
        """

        #
        coordsInPixels, gridShape = _computeGridPoints(length, self.display)
        nSubregions = coordsInPixels.shape[0]

        # Create the visual field
        field = visual.ElementArrayStim(
            self.display,
            fieldPos=coordsInPixels,
            fieldShape='sqr',
            nElements=nSubregions,
            sizes=length * self.display.ppd,
            elementMask=None,
            elementTex=None,
            units='pixels', 
        )

        #
        self._generateMetadata(
            nImagesPerBlock,
            nBlocksPerCondition,
            nBlockRepeats,
            includeJitteredBlocks,
            pSubregionHigh,
            nSubregions,
        )

        #
        offsetInPixels = np.full(2, round(length / 2 * self.display.ppd, 2)) * np.array(jitterDirection)
        self._runMainLoop(
            field,
            fieldCycle,
            nImagesPerBlock,
            nBlocksPerCondition,
            nBlockRepeats,
            includeJitteredBlocks,
            nSubregions,
            offsetInPixels,
            tIdle,
            nTrialsBetweenFlashes,
            flashCycle,
            nSignalFramesForField,
            nSignalFramesForFlash
        )

        #
        self.metadata['coords'] = np.around(coordsInPixels / self.display.ppd, 2)
        if correctVerticalReflection:
            self.metadata['coords'][:, 1] *= -1
        self.metadata['shape'] = gridShape
        self.metadata['length'] = length

        return
    
    def saveMetadata(self, sessionFolder):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        with open(sessionFolderPath.joinpath('binaryNoiseMetadata.pkl'), 'wb') as stream:
            pickle.dump(self.metadata, stream)

        return