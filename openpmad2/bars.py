import numpy as np
from itertools import product
from psychopy.visual import ShapeStim, ImageStim
from datetime import datetime as dt
from . import warping
from . import writing
from . import bases
from . import helpers
from . import constants

class MovingBarsMovie(bases.Stimulus):
    """
    """

    def __init__(
        self,
        display,
        width=30,
        orientations=[0, 45, 90],
        velocities=[50],
        repeats=1,
        margin=0,
        iti=0.2):
        """
        """

        super().__init__(display)

        self._width = width
        self._orientations = orientations
        self._velocities = velocities
        self._repeats = repeats
        self._margin = margin
        self._iti = iti

        return

    def construct(self, online=True, filename=None, metadata=None, prestimulus=1):
        """
        """

        if filename is not None:
                writer = writing.SKVideoVideoWriterWrapper(self.display, filename)
                self._movie = filename

        if metadata is not None:
            opened = open(metadata, 'w')
            # date, time = dt.now().strftime('%Y-%M-%d'), dt.now().strftime('%H:%m:%S')
            opened.write(f'Moving bar stimulus\n')
            opened.write(f'Bar width: {self.width} degrees\n')
            opened.write(f'Orientation, Direction, Velocity\n')

        #
        warper = warping.SignaledAndWarpedStim(self.display)
        bar = ShapeStim(self.display, fillColor='white', lineWidth=0, units='pix')

        #
        width = self.ppda * self.width
        length = np.sqrt(self.display.width ** 2 + self.display.width ** 2)
        vertices = np.array([
            [-1 * width / 2, -length / 2],
            [-1 * width / 2,  length / 2],
            [     width / 2,  length / 2],
            [     width / 2, -length / 2]
        ])
        bar.vertices = vertices

        #
        combos = np.repeat(
            np.array(list(product(self.orientations, self.velocities, [0, 1]))),
            self.repeats,
            axis=0
        )
        np.random.shuffle(combos)
        ntrials = len(combos)

        #
        distance = np.around(np.sqrt(
            (self.display.width + self.margin * 2) ** 2 + (self.display.width + self.margin * 2) ** 2
        ), 2).item() + width

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
        for itrial, (orientation, velocity, direction) in enumerate(combos):

            print(f'Working on trial {itrial + 1} out of {ntrials} ...')

            #
            if metadata is not None:
                opened.write(f'{orientation}, {direction}, {velocity}\n')

            #
            bar.pos = (0, 0) # Reset position to center of display
            ppf = velocity * self.ppda / self.display.fps
            bar.ori = orientation
            position = (
                np.cos(np.deg2rad(orientation)),
                np.sin(np.deg2rad(orientation))
            )
            bar.pos = position

            # Velocity and direction of motion
            remainder = distance % ppf
            offset = remainder / 2
            step = int(distance // ppf)
            radii = np.linspace(0 - offset, distance + offset, step) - (distance / 2) # Or should it be length / 2
            if direction:
                radii = radii[::-1]

            #
            counter = 0
            incoming = True
            outgoing = False
            warper.state = False
            for iframe, r in enumerate(radii):
                theta = np.deg2rad(180 - orientation) # Orientation of bar is orthoganol to the direction of motion
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                bar.pos = (x, y)
                bar.draw()
                image = self.display.getMovieFrame(buffer='back', clear=True)
                warper.array = image

                # Determine if the bar is in view
                if incoming:
                    if len(np.unique(warper.warped[warper.region])) > 1:
                        if counter == 0:
                            warper.state = True
                        elif counter == 2:
                            warper.state = False
                            outgoing = True # Change states
                            incoming = False # Change states
                            counter = 0 # Reset counter
                        if incoming:
                            counter += 1

                # Determine if bar is out of view
                if outgoing:
                    if len(np.unique(warper.warped[warper.region])) == 1:
                        if counter == 0:
                            warper.state = True
                        elif counter == 2:
                            warper.state = False
                            outgoing = False
                        counter += 1

                #
                warper.draw()
                warped = self.display.getMovieFrame(buffer='back', clear=False)

                #
                if filename is not None:
                    writer.write(warped)

                if online:
                    self.display.flip()
                else:
                    self.display.clearBuffer()

            # ITI
            self.display.clear()
            cleared = self.display.getMovieFrame()
            warper.array = cleared
            warper.draw()
            warped = self.display.getMovieFrame(clear=False)
            if online:
                self.display.flip()
            for iframe in range(int(np.ceil(self.display.fps * self.iti))):
                if filename is not None:
                    writer.write(warped)

        if filename is not None:
            writer.close()

        if metadata is not None:
            opened.close()

        return

    @property
    def width(self):
        return self._width

    @property
    def orientations(self):
        return self._orientations

    @property
    def velocities(self):
        return self._velocities

    @property
    def repeats(self):
        return self._repeats

    @property
    def margin(self):
        return self._margin

    @property
    def iti(self):
        return self._iti

class MovingBars():
    """
    """

    def __init__(
        self,
        display,
        width=90,
        velocity=50,
        orientations=[0, 45, 90],
        directions=[-1, 1],
        repeats=3,
        ):
        """
        """

        self.display = display
        self.width = width
        self.velocity = velocity
        self.orientations = orientations
        self.directions = directions
        self.repeats = repeats

        return

    def present(self):
        """
        """

        #
        combos = np.repeat(
            np.array(list(product(self.orientations, self.directions))),
            self.repeats,
            axis=0
        )
        np.random.shuffle(combos)
        ntrials = len(combos)

        #
        bar = ShapeStim(self.display, fillColor='white', lineWidth=0, units='pix')
        image = np.full([self.display.height, self.display.width], 0).astype(np.float64)
        background = ImageStim(
            self.display,
            image=image,
            size=self.display.size,
            units='pix',
        )

        #
        width = self.display.ppd * self.width
        hypotenuse = np.sqrt(self.display.width ** 2 + self.display.width ** 2)
        vertices = np.array([
            [-1 * width / 2, -1 * hypotenuse / 2],
            [-1 * width / 2,      hypotenuse / 2],
            [     width / 2,      hypotenuse / 2],
            [     width / 2, -1 * hypotenuse / 2]
        ])
        bar.vertices = vertices

        #
        ppf = self.velocity * self.display.ppd / self.display.fps
        diameter = hypotenuse + width
        remainder = diameter % ppf
        offset = remainder / 2
        step = int(diameter // ppf)
        steps = np.linspace(0 - offset, diameter + offset, step) - (diameter / 2)

        #
        self.display.wait(1)

        #
        for orientation, direction in combos:

            #
            if direction == -1:
                radii = steps[::-1]
            else:
                radii = steps

            #
            bar.ori = orientation
            theta = np.deg2rad(180 - orientation)

            #
            self.display.state = True
            for iframe, radius in enumerate(radii):
                if iframe == constants.N_SIGNAL_FRAMES:
                    self.display.state = False
                position = (
                    radius * np.cos(theta),
                    radius * np.sin(theta)
                )
                bar.pos = position
                background.draw()
                bar.draw()
                self.display.flip()

        return
