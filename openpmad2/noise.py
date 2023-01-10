from . import bases
from . import warping
from . import writing
from . import helpers
from . import constants
from psychopy import visual
import numpy as np
import pathlib as pl
from PIL import Image

class SparseNoise2D(bases.StimulusBase):
    """
    """

    def present(
        self,
        radius=5,
        duration=0.5,
        repeats=3,
        warmup=1
        ):
        """
        """

        self.header = {
            'Radius': f'{radius} degrees',
        }

        #
        spot = visual.Circle(
            self.display,
            fillColor='white',
            lineWidth=0,
            units='pix',
            size=2 * self.display.ppd * radius
        )

        #
        N = (
            int(self.display.azimuth // (radius * 2)),
            int(self.display.elevation // (radius * 2))
        )
        offset = (
            round(self.display.azimuth % (radius * 2) / 2, 2),
            round(self.display.elevation % (radius * 2) / 2, 2)
        )
        xi, yi = np.meshgrid(
            np.linspace(radius, N[0] * 2 * radius - radius, N[0]) - (self.display.azimuth / 2) + offset[0],
            np.linspace(radius, N[1] * 2 * radius - radius, N[1]) - (self.display.elevation / 2) + offset[1]
        )

        #
        xi = np.around(xi, 3)
        yi = np.around(yi, 3)
        coordsInDegrees = np.hstack([
            xi.reshape(-1, 1),
            yi.reshape(-1, 1)
        ])
        coordsInDegrees = np.repeat(coordsInDegrees, repeats, axis=0)
        coordsInPixels = coordsInDegrees * self.display.ppd

        #
        shuffledIndices = np.arange(coordsInPixels.shape[0])
        np.random.shuffle(shuffledIndices)
        coordsInPixels = coordsInPixels[shuffledIndices, :]
        coordsInDegrees = coordsInDegrees[shuffledIndices, :]

        #
        self.metadata = np.full([coordsInPixels.shape[0], 4], np.nan)
        self.metadata[:, :2] = coordsInDegrees

        #
        self.display.idle(warmup, units='seconds')

        #
        for trialIndex, coord in enumerate(coordsInPixels):

            #
            spot.pos = coord

            #
            self.display.signalEvent(3, units='frames')
            for frameIndex in range(int(np.ceil(self.display.fps * duration))):
                self.display.drawBackground()
                spot.draw()
                if frameIndex == 0:
                    self.metadata[trialIndex, 2] = self.display.flip()
                else:
                    self.display.flip()

            #
            self.display.signalEvent(3, units='frames')
            for frameIndex in range(int(np.ceil(self.display.fps * duration))):
                self.display.drawBackground()
                if frameIndex == 0:
                    self.metadata[trialIndex, 3] = self.display.flip()
                else:
                    self.display.flip()

        return

    def saveMetadata(self, sessionFolder):
        """
        """

        self.header.update({'Columns': 'Azimuth (degrees), Elevation (degrees), Timestamp (On), Timestamp (Off)'})
        stream = super().prepareMetadataStream(sessionFolder, 'sparseNoiseMetadata', self.header)
        for x, y, t1, t2 in self.metadata:
            line = f'{x:.1f}, {y:.1f}, {t1:.3f}, {t2:.3f}\n'
            stream.write(line)
        stream.close()

        return

class SuperResolutionLowFrequencyBinaryNoise(bases.StimulusBase):
    """
    """

    def present(
        self,
        length=5,
        repeats=1,
        duration=10,
        cycle=(0.5, 0.5),
        probability=0.2,
        randomize=True,
        ):
        """
        """

        shiftInDegrees = round(length / 2, 2)
        shiftInPixels = shiftInDegrees * self.display.ppd

        #
        N = (
            int(self.display.azimuth // length),
            int(self.display.elevation // length)
        )
        offset = (
            round(self.display.azimuth % length / 2, 2),
            round(self.display.elevation % length / 2, 2)
        )
        xi, yi = np.meshgrid(
            np.linspace(length, N[0] * length - (length / 2), N[0]) - (self.display.azimuth / 2) + offset[0],
            np.linspace(length, N[1] * length - (length / 2), N[1]) - (self.display.elevation / 2) + offset[1]
        )

        #
        xi = np.around(xi, 3)
        yi = np.around(yi, 3)
        coordsInDegrees = np.hstack([
            xi.reshape(-1, 1),
            yi.reshape(-1, 1)
        ])
        coordsInDegrees = np.repeat(coordsInDegrees, repeats, axis=0)
        coordsInPixels = coordsInDegrees * self.display.ppd
        nSubregions = coordsInDegrees.shape[0]

        #
        field = [
            visual.Rect(self.display, units='pix')
                for i in range(nSubregions)
        ]
        for subregion, (x, y) in zip(field, coordsInPixels):
            subregion.pos = x, y
            subregion.height = self.display.ppd * length
            subregion.width = self.display.ppd * length
            subregion.lineColor = 0
            subregion.lineWidth = 0.0

        #
        tBlock = 2 * np.sum(cycle)
        nBlocks = duration // tBlock
        nTrials = int(nBlocks * 2)

        #
        self.metadata = {
            'M': np.full([nTrials, nSubregions], False, dtype=bool), # Masks
            'C': np.full([nTrials, nSubregions, 2], np.nan), # coordinates
            'I': np.full([nTrials, self.display.height, self.display.width], np.nan), # Images
        }

        # Generate the trial sequence
        iTrial = 0
        while iTrial < nTrials:

            # Create a mask
            mask = np.random.choice(
                a=[True, False],
                size=nSubregions,
                p=[probability, 1 - probability],
            ).astype(bool)

            # Iterate through each phase: original/shifted
            for phase in ('original', 'shifted'):

                # Compute the spatial offset
                if phase == 'original':
                    offset = np.array([0, 0])
                else:
                    offset = np.array([0, 0]) + shiftInPixels
                    # theta = np.random.choice([45, 135, 225, 315], size=1).item()
                    # offset = np.array([
                    #     shiftInPixels * np.cos(np.deg2rad(theta)),
                    #     shiftInPixels * np.sin(np.deg2rad(theta))
                    # ]) * self.display.ppd
                coords = np.around(coordsInPixels + offset, 2)

                #
                self.metadata['M'][iTrial] = mask
                self.metadata['C'][iTrial] = coords

                #
                iTrial += 1

        # Randomize the trials
        if randomize:
            shuffle = np.arange(nTrials)
            np.random.shuffle(shuffle)
            for key in ('M', 'C'):
                self.metadata[key] = self.metadata[key][shuffle]

        # Change the background to black and wait 5 seconds
        if self.display.backgroundColor != -1:
            self.display.backgroundColor = -1
            self.display.idle(5, units='seconds')

        # Main presentation loop
        for iTrial in range(nTrials):

            #
            self.display.clearBuffer()

            #
            mask = self.metadata['M'][iTrial]
            coords = self.metadata['C'][iTrial]

            # Set a new field pattern
            for flag, (x, y), subregion in zip(mask, coords, field):

                #
                color = 1 if flag else -1
                subregion.fillColor = color
                subregion.pos = (x, y)
                subregion.draw()

            #
            # image = np.array(
            #     self.display.getMovieFrame(buffer='back').convert('L').transpose(Image.FLIP_TOP_BOTTOM)
            # )
            # self.metadata['I'][iTrial] = image

            #
            self.display.signalEvent(1, units='frames')
            for iFrame in range(int(np.ceil(cycle[0] * self.display.fps))):
                self.display.flip(clearBuffer=False)

            #
            self.display.signalEvent(1, units='frames')
            self.display.drawBackground()
            for iFrame in range(int(np.ceil(cycle[1] * self.display.fps))):
                self.display.flip(clearBuffer=False)

        return

    def saveMetadata(self, sessionFolder):
        """
        TODO: Code this
        """

        # NOTE: Images are flipped vertically through the projector.
        #       Y-values need to be multiplied by -1.

        return