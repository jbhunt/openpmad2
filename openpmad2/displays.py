from cgitb import text
import numpy as np
from psychopy.visual import Window
from psychopy.visual.windowwarp import Warper
from psychopy.visual import GratingStim, ImageStim
from . import warping

_lowStateTexture = np.full([16, 16], -1).astype(np.int8)
_highStateTexture = np.full([16, 16], 1).astype(np.int8)

class WarpedWindow(Window):
    """
    """

    def __init__(
        self,
        size=(1280, 720),
        fov=(180, 100),
        fps=60,
        screen=1,
        color=0,
        fullScreen=False,
        patchCoords=(-7, 345, 40, 66),
        textureShape=(16, 16)
        ):
        """
        """

        #
        self._countdown = None
        self._width, self._height = size
        self._azimuth, self._elevation = fov
        self._fps = fps
        self._ppd = np.mean([
            self.width / self.azimuth,
            self.height / self.elevation
        ])
        self._textureShape = textureShape

        #
        super().__init__(
            size=(self.width, self.height),
            screen=screen,
            units='pix',
            gammaErrorPolicy='warn',
            useFBO=True,
            color=-1,
            checkTiming=False,
            fullscr=fullScreen,
        )

        self._warper = Warper(self, warp='warpfile', warpfile=warping.WARPFILE)

        #
        self._state = False
        self._patchCoords = patchCoords
        x, y, w, h = self._patchCoords
        self._patch = GratingStim(
            self,
            tex=_lowStateTexture,
            size=(w, h),
            pos=(x, y),
            units='pix'
        )

        #
        self._backgroundColor = color
        self._background = GratingStim(
            self,
            tex=np.full(self._textureShape, color),
            size=(self.width, self.height),
            units='pix'
        )

        self._background.draw()
        self.flip()

        return

    def flip(self, drawSignalPatch=True, **kwargs):
        """
        """

        # Draw the signal patch
        if self._countdown is not None:
            if self._countdown != 0:
                self._countdown -= 1
            else:
                if self._state != False:
                    self._patch.tex = _lowStateTexture
                    self._state = False
                    self._countdown = None
        if drawSignalPatch:
            self._patch.draw()

        return super().flip(**kwargs)

    def idle(self, duration=1, units='seconds', returnFirstTimestamp=False):
        """
        """

        if units == 'frames':
            frameCount = int(duration)
        elif units == 'seconds':
            frameCount = round(self.fps * duration)

        for frameIndex in range(frameCount):
            self._background.draw()
            if frameIndex == 0:
                timestamp = self.flip()
            else:
                self.flip()

        if returnFirstTimestamp:
            return timestamp

    def signalEvent(self, duration=3, units='frames'):
        """
        Flash the visual patch for a specific amount of time

        keywords
        --------
        duration: int of float
            Duration of the signalling event
        units: str
            Unit of time (frames or seconds)
        """

        self._patch.tex = _highStateTexture
        self._state = True
        if units == 'frames':
            self._countdown = int(duration)
        elif units == 'seconds':
            self._countdown = round(self.fps * duration)
        else:
            raise Exception(f'{units} is an invalid unit of time')

        return

    def clearStimuli(self):
        """
        """

        self._background.draw()
        timestamp = self.flip()

        return timestamp

    def drawBackground(self):
        """
        """

        self._background.draw()

        return

    def getNumpyArray(self, buffer='back'):
        """
        """

        return np.array(self.getMovieFrame(buffer=buffer).convert('L'))

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def azimuth(self):
        return self._azimuth

    @property
    def elevation(self):
        return self._elevation

    @property
    def fps(self):
        return self._fps

    @property
    def ppd(self):
        return self._ppd

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        """
        """

        if value not in [True, False]:
            raise Exception(f'Invalid state: {value}')
        if value:
            self._patch.tex = _highStateTexture
        else:
            self._patch.tex = _lowStateTexture
        self._state = True if value else False

        return

    @property
    def patchCoords(self):
        return np.concatenate([self._patch.pos, self._patch.size])

    @patchCoords.setter
    def patchCoords(self, coords):
        x, y, w, h = coords
        self._patch.pos = (x, y)
        self._patch.size = (w, h)
        self._background.draw()
        self.flip()

    @property
    def backgroundColor(self):
        return self._backgroundColor

    @backgroundColor.setter
    def backgroundColor(self, color):
        if color < -1 or color > 1:
            raise Exception('Background color must be in the range (-1, 1)')
        self._background.tex = np.full(self._textureShape, color)
        self._backgroundColor = color
        self._background.draw()
        self.flip()