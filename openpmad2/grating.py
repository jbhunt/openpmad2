import numpy as np
import pathlib as pl
from itertools import product
from psychopy.visual import GratingStim, ImageStim
from datetime import datetime as dt
from . import bases
from . import warping
from . import writing
from . import constants

class OrientedDriftingGratingMovie(bases.Stimulus):
    """
    """

    def __init__(
        self,
        display,
        frequency=0.1, # cycles / degree
        velocity=25, # degrees / second
        orientations=[45], # degrees
        duration=3,
        repeats=1,
        static=0.5, # seconds
        iti=1, # seconds
        ):
        """
        """

        super().__init__(display)

        self._frequency = frequency
        self._orientations = orientations
        self._duration = duration
        self._velocity = velocity
        self._repeats = repeats
        self._static = static
        self._iti = iti

        return

    def construct(self, online=True, filename=None, metadata=None, prestimulus=1):
        """
        """

        #
        if filename is not None:
            writer = writing.SKVideoVideoWriterWrapper(self.display, filename)
            self._movie = filename

        if metadata is not None:
            opened = open(metadata, 'w')
            opened.write(f'Spatial frequency: {self.frequency} cycles / degree\n')
            opened.write(f'Velocity: {self.velocity} degrees / second\n')
            opened.write(f'Motion duration: {self.duration}\n')
            opened.write(f'Orientation, Direction\n')

        #
        cpp = self.frequency / self.ppda # cycles per pixel
        cpf = self.frequency * self.velocity / self.display.fps #
        ppc = 1 / cpp

        #
        warper = warping.SignaledAndWarpedStim(self.display)
        grating = GratingStim(self.display, units='pix')
        hypotenuse = np.sqrt(self.display.width ** 2 + self.display.width ** 2)
        grating.size = (hypotenuse, hypotenuse)
        grating.sf = cpp

        #
        combos = np.repeat(
            np.array(list(product(self.orientations, [-1, +1]))),
            self.repeats,
            axis=0
        )
        np.random.shuffle(combos)

        # Short pre-stimulus epoch
        self.display.background.state = False
        self.display.background.draw()
        warped = self.display.getMovieFrame()
        for iframe in range(int(np.ceil(prestimulus * self.display.fps))):
            if online:
                self.display.background.draw()
                self.display.flip()
            if filename is not None:
                writer.write(warped)

        #
        for orientation, direction in combos:

            if metadata is not None:
                human_readable_direction = 'Left' if direction == -1 else 'Right'
                opened.write(f'{orientation}, {human_readable_direction}\n')

            # Set orientation
            grating.ori = orientation

            # Present grating static for 1 second
            grating.draw()
            frame = self.display.getMovieFrame()
            warper.array = frame
            warper.state = True
            for iframe in range(int(np.ceil(self.static * self.display.fps))):
                if iframe == 2:
                    warper.state = False
                warper.draw()
                warped = self.display.getMovieFrame(clear=False)
                if online:
                    self.display.flip()
                if filename is not None:
                    writer.write(warped)

            # Present grating motion
            grating.phase = 0
            warper.state = True
            for iframe in range(int(np.ceil(self.duration * self.display.fps))):
                if iframe == 2:
                    warper.state = False
                grating.phase = (iframe + 1) * cpf * direction
                grating.draw()
                frame = self.display.getMovieFrame()
                warper.array = frame
                warper.draw()
                warped = self.display.getMovieFrame(clear=False)
                if filename is not None:
                    writer.write(warped)
                if online:
                    self.display.flip()

            # ITI

            # Signal on
            self.display.background.state = True
            self.display.background.draw()
            warped = self.display.getMovieFrame()
            for iframe in range(2):
                if online:
                    self.display.background.draw()
                    self.display.flip()
                if filename is not None:
                    writer.write(warped)

            # Signal off
            self.display.background.state = False
            self.display.background.draw()
            warped = self.display.getMovieFrame()
            for iframe in range(int(np.ceil(self.iti * self.display.fps)) - 2):
                if online:
                    self.display.background.draw()
                    self.display.flip()
                if filename is not None:
                    writer.write(warped)

        if filename is not None:
            writer.close()

        if metadata is not None:
            opened.close()

        return

    @property
    def frequency(self):
        return self._frequency

    @property
    def orientations(self):
        return self._orientations

    @property
    def velocity(self):
        return self._velocity

    @property
    def repeats(self):
        return self._repeats

    @property
    def duration(self):
        return self._duration

    @property
    def static(self):
        return self._static

    @property
    def iti(self):
        return self._iti

class OrientedDriftingGrating():
    """
    """

    def __init__(
        self,
        display,
        frequency=0.08, # cycles / degree
        velocity=50, # degrees / second
        orientations=[45], # degrees
        duration=2,
        repeats=3,
        tstatic=1, # seconds
        iti=1, # seconds
        ):
        """
        """

        self._display = display
        self._frequency = frequency
        self._orientations = orientations
        self._duration = duration
        self._velocity = velocity
        self._repeats = repeats
        self._tstatic = tstatic
        self._iti = iti

        return

    def present(self, warmup=1):
        """
        """

        #
        cpp = self.frequency / self.display.ppd # cycles per pixel
        cpf = self.frequency * self.velocity / self.display.fps #
        ppc = 1 / cpp

        #
        grating = GratingStim(self.display, units='pix')
        hypotenuse = np.sqrt(self.display.width ** 2 + self.display.width ** 2)
        grating.size = (hypotenuse, hypotenuse)
        grating.sf = cpp

        #
        image = np.full([self.display.height, self.display.width], 0).astype(np.float64)
        background = ImageStim(
            self.display,
            image=image,
            size=self.display.size,
            units='pix',
        )

        #
        combos = np.repeat(
            np.array(list(product(self.orientations, [-1, 1]))),
            self.repeats,
            axis=0
        )
        np.random.shuffle(combos)

        # Pre-stimulus epoch
        self.display.wait(warmup)

        #
        for orientation, direction in combos:

            #
            grating.ori = orientation

            #
            self.display.state = True
            for iframe in range(int(np.ceil(self.display.fps * self.tstatic))):
                if iframe == constants.N_SIGNAL_FRAMES:
                    self.display.state = False
                grating.draw()
                self.display.flip()

            #
            self.display.state = True
            for iframe in range(int(np.ceil(self.display.fps * self.duration))):
                if iframe == constants.N_SIGNAL_FRAMES:
                    self.display.state = False
                grating.phase += direction * cpf
                grating.draw()
                self.display.flip()

            #
            self.display.state = True
            for iframe in range(int(np.ceil(self.display.fps * self.iti))):
                if iframe == constants.N_SIGNAL_FRAMES:
                    self.display.state = False
                background.draw()
                self.display.flip()

        return

    @property
    def display(self):
        return self._display

    @property
    def frequency(self):
        return self._frequency

    @property
    def orientations(self):
        return self._orientations

    @property
    def velocity(self):
        return self._velocity

    @property
    def repeats(self):
        return self._repeats

    @property
    def duration(self):
        return self._duration

    @property
    def tstatic(self):
        return self._tstatic

    @property
    def iti(self):
        return self._iti
