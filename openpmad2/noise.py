from requests import session
from . import bases
from . import warping
from . import writing
from . import helpers
from . import constants
from psychopy import visual
import numpy as np
import pathlib as pl

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
