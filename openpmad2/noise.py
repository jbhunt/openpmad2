from . import bases
from . import warping
from . import writing
from . import helpers
from . import constants
from psychopy import visual
import numpy as np

class SparseNoise2DMovie(bases.Stimulus):
    """
    """

    def __init__(
        self,
        display,
        radius=10,
        duration=0.5,
        repeats=3,
        ):
        """
        """

        super().__init__(display)
        self.radius = radius
        self.duration = duration
        self.repeats = repeats
        # self.movie = None

        #
        if self.display.azimuth % radius != 0 or self.display.elevation % radius != 0:
            raise Exception(f'Display size must be evenly divisible by spot size')

        return

    def construct(self, filename=None, metadata=None, online=True):
        """
        """

        if filename is not None:
            writer = writing.SKVideoVideoWriterWrapper(self.display, filename)

        if metadata is not None:
            opened = open(metadata, 'w')
            opened.write(f'Stimulus: Sparse noise (2D)\n')
            opened.write(f'Radius: {self.radius} degrees\n')
            opened.write(f'Repeats: {self.repeats}\n')
            opened.write(f'Duration: {self.duration} seconds\n')
            opened.write(f'----------\n')

        # Init stimuli
        radius = self.ppda * self.radius
        spot = visual.Circle(
            self.display,
            fillColor='white',
            lineWidth=0,
            units='pix'
        )
        spot.size = radius * 2
        warper = warping.SignaledAndWarpedStim(self.display)

        #
        xi, yi = np.meshgrid(
            np.arange(-1 * self.display.azimuth / 2, self.display.azimuth / 2, self.radius * 2),
            np.arange(-1 * self.display.elevation / 2, self.display.elevation / 2, self.radius * 2)
        )
        xi = (xi + self.radius) * self.ppda
        yi = (yi + self.radius) * self.ppda
        stacked = np.hstack([
            xi.reshape(-1, 1),
            yi.reshape(-1, 1)
        ])
        coords = np.repeat(stacked, self.repeats, axis=0)
        np.random.shuffle(coords)

        # Pre-stimulus period
        self.display.showBlankScreen(t=1, writer=writer)

        #
        for coord in coords:

            #
            if metadata is not None:
                x, y = np.around(coord / self.ppda, 0).astype(int) # Round to the nearest whole degree
                opened.write(f'{x}, {y}\n')

            #
            spot.pos = coord
            spot.draw()
            image = self.display.getMovieFrame(buffer='back')
            warper.array = image

            #
            warper.state = True
            warper.draw()
            image = self.display.getMovieFrame(buffer='back', clear=False)
            if online:
                self.display.flip()

            #
            for iframe in range(int(np.ceil(self.display.fps * self.duration))):

                #
                if iframe == 2:
                    warper.state = False
                    warper.draw()
                    image = self.display.getMovieFrame(buffer='back', clear=False)

                    #
                    if online:
                        self.display.flip()

                #
                if filename is not None:
                    writer.write(image)

            #
            self.display.background.state = True
            self.display.background.draw()
            image = self.display.getMovieFrame(buffer='back', clear=False)
            if online:
                self.display.flip()

            #
            for iframe in range(int(np.ceil(self.display.fps * self.duration))):

                #
                if iframe == 2:
                    self.display.background.state = False
                    self.display.background.draw()
                    image = self.display.getMovieFrame(buffer='back', clear=False)

                    #
                    if online:
                        self.display.flip()

                #
                if filename is not None:
                    writer.write(image)

        if filename is not None:
            writer.close()

        if metadata is not None:
            opened.close()

        return

    def play(self):
        """
        """

        return

class SparseNoise2D():
    """
    """

    def __init__(
        self,
        display,
        radius=10,
        duration=0.5,
        repeats=3
        ):
        """
        """

        self.display = display
        self.radius = radius
        self.duration = duration
        self.repeats = repeats

        return

    def present(self, warmup=1):
        """
        """

        #
        radius = self.display.ppd * self.radius
        spot = visual.Circle(
            self.display,
            fillColor='white',
            lineWidth=0,
            units='pix',
            size=self.display.ppd * self.radius * 2
        )

        #
        # alpha = helpers.make_alpha_mask(margin=30, sigma=10, low=-1, high=0)
        image = np.full([self.display.height, self.display.width], 0).astype(np.float64)
        background = visual.ImageStim(
            self.display,
            image=image,
            size=self.display.size,
            units='pix',
        )

        #
        xi, yi = np.meshgrid(
            np.arange(-1 * self.display.azimuth / 2, self.display.azimuth / 2, self.radius * 2),
            np.arange(-1 * self.display.elevation / 2, self.display.elevation / 2, self.radius * 2)
        )
        xi = (xi + self.radius) * self.display.ppd
        yi = (yi + self.radius) * self.display.ppd
        stacked = np.hstack([
            xi.reshape(-1, 1),
            yi.reshape(-1, 1)
        ])
        coords = np.repeat(stacked, self.repeats, axis=0)
        np.random.shuffle(coords)

        #
        self.display.wait(warmup)

        #
        for coord in coords:

            #
            spot.pos = coord

            #
            self.display.state = True
            for iframe in range(int(np.ceil(self.display.fps * self.duration))):
                if iframe == constants.N_SIGNAL_FRAMES:
                    self.display.state = False
                background.draw()
                spot.draw()
                self.display.flip()

            #
            self.display.state = True
            for iframe in range(int(np.ceil(self.display.fps * self.duration))):
                if iframe == constants.N_SIGNAL_FRAMES:
                    self.display.state = False
                background.draw()
                self.display.flip()


        return
