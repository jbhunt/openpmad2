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
    xi = np.around(xi + offset[0] - (display.width / 2), 2)
    yi = np.around(yi + offset[1] - (display.height / 2), 2)

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
        randomize
        ):
        """
        """

        #
        coordsInDegrees = np.around(coordsInPixels / self.display.ppd, 3)
        gridHeight, gridWidth = gridShape
        self.metadata = {
            'fields': np.full([nTrials, gridHeight, gridWidth], np.nan),
            'coords': np.full([nTrials, 2], np.nan),
            'indices': np.full([nTrials, 1], 0).astype(np.int64)
        }

        #
        iTrial = 0
        for iRepeat in range(repeats):
            for iSubregion in np.arange(field.nElements):
                image = np.full(field.nElements, -1)
                image[iSubregion] = 1
                self.metadata['fields'][iTrial] = image.reshape(gridShape)
                self.metadata['coords'][iTrial] = coordsInDegrees[iSubregion]
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
        ):
        """
        """

        if self.display.backgroundColor != -1:
            self.display.backgroundColor = -1
        self.display.idle(tIdle)

        for iTrial in range(nTrials):

            #
            iSubregion = self.metadata['indices'][iTrial]
            colors = np.full(field.nElements, -1)
            colors[iSubregion] = 1
            field.colors = colors.reshape(-1, 1)
            
            #
            self.display.signalEvent(1, units='frames')
            for iFrame in range(int(np.ceil(cycle[0] * self.display.fps))):
                field.draw()
                self.display.flip()

            #
            colors = np.full(field.nElements, -1).reshape(-1, 1)
            field.colors = colors

            #
            self.display.signalEvent(1, units='frames')
            for iFrame in range(int(np.ceil(cycle[1] * self.display.fps))):
                field.draw()
                self.display.flip()

        self.display.idle(tIdle)

        return

    def present(
        self,
        radius=5,
        cycle=(0.5, 0.5),
        repeats=1,
        tIdle=3,
        randomize=True,
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
            elementMask=None,
            elementTex=None,
            units='pixels', 
        )

        # Generate metadata
        self._generateMetadata(coordsInPixels, repeats, field, nTrials, gridShape, randomize)
        self.metadata['length'] = length
        self.metadata['cycle'] = cycle

        # Run main loop
        self._runMainLoop(field, tIdle, cycle, nTrials)

        return

    def saveMetadata(self, sessionFolder, correctVerticalReflection=False):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        #
        output = self.metadata.copy()
        if correctVerticalReflection:
            output['fields'] = np.flip(self.metadata['fields'], axis=1)
            output['coords'][:, 1] = -1 * self.metadata['coords'][:, :, 1]

        #
        with open(sessionFolderPath.joinpath('binaryNoiseMetadata.pkl'), 'wb') as stream:
            pickle.dump(output, stream)

        return    

class BinaryNoise(bases.StimulusBase):
    """
    """

    def present(
        self,
        length=5,
        tImage=0.05,
        tPresent=3,
        pHigh=0.2,
        tIdle=3,
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

        # Change the background to black and wait 5 seconds
        if self.display.backgroundColor != -1:
            self.display.backgroundColor = -1
        self.display.idle(tIdle, units='seconds')

        # Populate the metadata dictionary
        nTrials = int(tPresent // tImage)
        self.metadata = {
            'fields': np.full([nTrials, gridHeight, gridWidth], 0).astype(np.float64),
            'coords': np.full([nTrials, nSubregions, 2], np.nan),
            'values': np.full([nTrials, nSubregions], np.nan)
        }
        self.metadata['length'] = length
        self.metadata['interval'] = tImage

        #
        for iTrial in range(nTrials):

            #
            colors = np.random.choice(
                a=np.array([-1, 1]),
                p=np.array([1 - pHigh, pHigh]),
                size=nSubregions
            ).reshape(-1, 1)
            field.colors = colors

            #
            self.metadata['coords'][iTrial, :, 0] = coordsInDegrees[:, 0]
            self.metadata['coords'][iTrial, :, 1] = coordsInDegrees[:, 1]

            #
            self.display.signalEvent(1, units='frames')
            for iFrame in range(int(np.ceil(tImage * self.display.fps))):
                field.draw()
                self.metadata['fields'][iTrial] = colors.reshape(gridHeight, gridWidth)
                self.display.flip()

        # Display black screen for 5 seconds
        self.display.idle(tIdle, units='seconds')

        return

    def saveMetadata(self, sessionFolder, correctVerticalReflection=True):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        #
        output = self.metadata.copy()
        if correctVerticalReflection:
            output['fields'] = np.flip(self.metadata['fields'], axis=1)
            output['coords'][:, :, 1] = -1 * self.metadata['coords'][:, :, 1]

        #
        with open(sessionFolderPath.joinpath('binaryNoiseMetadata.pkl'), 'wb') as stream:
            pickle.dump(output, stream)

        return

class JitteredNoise(bases.StimulusBase):
    """
    """

    def _generateMetadata(
        self,
        cycle,
        duration,
        probability,
        nSubregions,
        shiftInPixels,
        coordsInPixels,
        gridShape,
        ):
        """
        """

        #
        tBlock = 2 * np.sum(cycle)
        nBlocks = duration // tBlock
        nTrials = int(nBlocks * 2)

        #
        gridHeight, gridWidth = gridShape
        self.metadata = {
            'fields': np.full([nTrials, gridHeight, gridWidth], np.nan).astype(np.float64),
            'coords': np.full([nTrials, nSubregions, 2], np.nan), # coordinates (in pixels)
        }

        # Generate the trial sequence
        iTrial = 0
        while iTrial < nTrials:

            # Create a mask
            values = np.random.choice(
                a=[1, -1],
                size=gridShape,
                p=[probability, 1 - probability],
            )

            # Iterate through each phase: original/shifted
            for phase in ('original', 'shifted'):

                # Compute the spatial offset
                if phase == 'original':
                    offset = np.array([0, 0])
                else:
                    offset = np.random.choice([-1, 1], size=2) * (np.array([0, 0]) + shiftInPixels)
                coords = np.around(coordsInPixels + offset, 2)

                #
                self.metadata['coords'][iTrial] = coords
                self.metadata['fields'][iTrial] = values

                #
                iTrial += 1

        return

    def _runMainLoop(
        self,
        nTrials,
        field,
        cycle,
        tIdle,
        ):
        """
        """

        self.display.idle(tIdle, units='seconds')

        #
        for iTrial in range(nTrials):

            #
            self.display.clearBuffer()

            # Set a new field pattern
            field.fieldPos = self.metadata['coords'][iTrial]
            field.colors = self.metadata['fields'][iTrial].reshape(field.nElements, 1)
            field.draw()

            # Save a copy of the stimulus
            image = np.array(
                self.display.getMovieFrame(buffer='back').convert('L').transpose(Image.FLIP_TOP_BOTTOM)
            )

            # Present the field
            self.display.signalEvent(1, units='frames')
            for iFrame in range(int(np.ceil(cycle[0] * self.display.fps))):
                self.display.flip(clearBuffer=False)

            # ITI
            self.display.signalEvent(1, units='frames')
            self.display.drawBackground()
            for iFrame in range(int(np.ceil(cycle[1] * self.display.fps))):
                self.display.flip(clearBuffer=False)

        self.display.idle(tIdle, units='seconds')

        return

    def present(
        self,
        length=10,
        repeats=1, # TODO: Implement this
        duration=10,
        cycle=(0.5, 0.5),
        probability=0.2,
        randomize=True,
        tIdle=3,
        ):
        """
        """

        #
        coordsInPixels, gridShape = _computeGridPoints(length, self.display)
        nSubregions = coordsInPixels.shape[0]
        initialColors = np.random.choice([-1, 1], p=[1 - probability, probability], size=nSubregions).reshape(-1, 1)

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
        shiftInPixels = round(length / 2 * self.display.ppd, 2)
        self._generateMetadata(
            cycle,
            duration,
            probability,
            nSubregions,
            shiftInPixels,
            coordsInPixels,
            gridShape
        )

        # Determine the total number of trials
        tBlock = 2 * np.sum(cycle)
        nBlocks = duration // tBlock
        nTrials = int(nBlocks * 2)

        # Randomize the trials
        if randomize:
            shuffle = np.arange(nTrials)
            np.random.shuffle(shuffle)
            for key in ('fields', 'coords'):
                self.metadata[key] = self.metadata[key][shuffle]

        # Change the background to black and wait 5 seconds
        if self.display.backgroundColor != -1:
            self.display.backgroundColor = -1

        # Run main presentation loop
        self._runMainLoop(
            nTrials,
            field,
            cycle,
            tIdle
        )

        return

    def saveMetadata(self, sessionFolder, correctVerticalReflection=False):
        """
        """

        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()


        #
        output = self.metadata.copy()
        if correctVerticalReflection:
            output['fields'] = np.flip(self.metadata['fields'], axis=1)
            output['coords'][:, :, 1] = -1 * self.metadata['coords'][:, :, 1]

        #
        with open(sessionFolderPath.joinpath('jitteredNoiseMetadata.pkl'), 'wb') as stream:
            pickle.dump(self.metadata, stream)

        return