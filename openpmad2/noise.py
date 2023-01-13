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
            'indices': np.full([nTrials, 1], 0).astype(np.int64)
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
            self.display.signalEvent(3, units='frames')
            for iFrame in range(int(np.ceil(cycle[0] * self.display.fps))):
                field.draw()
                self.display.flip()

            #
            colors = np.full(field.nElements, -1).reshape(-1, 1)
            field.colors = colors

            #
            self.display.signalEvent(3, units='frames')
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
        correctVerticalReflection=True
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
        self._runMainLoop(field, tIdle, cycle, nTrials)

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
        correctVerticalReflection=True,
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
            'coords': coordsInDegrees,
        }
        if correctVerticalReflection:
            self.metadata['coords'][:, 1] = self.metadata['coords'][:, 1] * -1
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
            self.display.signalEvent(3, units='frames')
            for iFrame in range(int(np.ceil(tImage * self.display.fps))):
                field.draw()
                self.metadata['fields'][iTrial] = colors.reshape(gridHeight, gridWidth)
                self.display.flip()

        # Display black screen for 5 seconds
        self.display.idle(tIdle, units='seconds')

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

class JitteredNoise(bases.StimulusBase):
    """
    """

    def _generateMetadata(
        self,
        nImages,
        repeats,
        probability,
        nSubregions,
        shiftInPixels,
        coordsInPixels,
        gridShape,
        randomize,
        ):
        """
        """

        #
        nTrials = nImages * 2 * repeats
        gridHeight, gridWidth = gridShape
        self.metadata = {

            # Values for each of the subregions on the ith trial
            'fields': np.full([nTrials, gridHeight, gridWidth], np.nan).astype(np.float64),

            # x and y coordinates (in pixels) for each of the subregions on the ith trial
            'coords': np.full([nTrials, nSubregions, 2], np.nan),

            # x and y offsets (in degrees) for each subregion on the ith trial
            'offsets': np.full([nTrials, 2], np.nan),

            #
            'shifted': np.full([nTrials, 1], False),

            #
            'masks': np.full([nTrials, nSubregions, 1], False)
        }

        # Generate the trial sequence
        iTrial = 0
        for iBlock in range(nImages):

            # Generate a new image
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
                for iRepeat in range(repeats):

                    #
                    self.metadata['coords'][iTrial] = coords
                    self.metadata['fields'][iTrial] = values
                    self.metadata['offsets'][iTrial] = np.around(offset / self.display.ppd, 2)
                    self.metadata['shifted'][iTrial] = True if phase == 'shifted' else False
                    self.metadata['masks'][iTrial] = np.array(values == 1).reshape(-1, 1)
                    iTrial += 1

        # Randomize the sequence of images    
        if randomize:
            shuffle = np.arange(nTrials)
            np.random.shuffle(shuffle)
            for key in ('fields', 'coords', 'offsets', 'shifted', 'masks'):
                self.metadata[key] = self.metadata[key][shuffle]    

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

        # Change the background to black and idle
        if self.display.backgroundColor != -1:
            self.display.backgroundColor = -1
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
            self.display.signalEvent(3, units='frames')
            for iFrame in range(int(np.ceil(cycle[0] * self.display.fps))):
                self.display.flip(clearBuffer=False)

            # ITI
            nFramesITI = int(np.ceil(cycle[1] * self.display.fps))
            if nFramesITI == 0:
                continue
            self.display.signalEvent(3, units='frames')
            self.display.drawBackground()
            for iFrame in range(nFramesITI):
                self.display.flip(clearBuffer=False)

        self.display.idle(tIdle, units='seconds')

        return

    def present(
        self,
        length=10,
        repeats=1,
        nImages=10,
        cycle=(0.5, 0.5),
        probability=0.2,
        randomize=True,
        tIdle=3,
        correctVerticalReflection=True,
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
            nImages,
            repeats,
            probability,
            nSubregions,
            shiftInPixels,
            coordsInPixels,
            gridShape,
            randomize
        )

        # Determine the total number of trials
        nTrials = self.metadata['fields'].shape[0]

        # Run main presentation loop
        self._runMainLoop(
            nTrials,
            field,
            cycle,
            tIdle
        )

        # Process metadata before saving
        self.metadata['length'] = length
        self.metadata['cycle'] = cycle
        self.metadata['coords'] = np.around(self.metadata['coords'] / self.display.ppd, 2)
        if correctVerticalReflection:
            self.metadata['coords'][:, :, 1] = -1 * self.metadata['coords'][:, :, 1] # Reflect the coords over the horizontal axis
            self.metadata['offsets'][:, 1] = self.metadata['offsets'][:, 1] * -1 # Reflect the offsets over the horizontal axis

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