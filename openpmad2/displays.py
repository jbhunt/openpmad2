import numpy as np
from psychopy.visual import Window
from psychopy.visual.windowwarp import Warper
from psychopy.visual import GratingStim, ImageStim
from . import warping

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
        patch=(30, 30, 0, 0)
        ):
        """
        """

        #
        self.signalFrameCount = 0
        self._width, self._height = size
        self._azimuth, self._elevation = fov
        self._fps = fps
        self._ppd = np.mean([
            self.width / self.azimuth,
            self.height / self.elevation
        ])

        #
        super().__init__(
            size=(self.width, self.height),
            screen=screen,
            units='pix',
            gammaErrorPolicy='warn',
            useFBO=True,
            color=-1,
            checkTiming=False,
        )

        self._warper = Warper(self, warp='warpfile', warpfile=warping.WARPFILE)

        #
        self._state = False
        w, h, x, y = patch
        self._patch = GratingStim(
            self,
            tex=np.full([16, 16], -1),
            size=(w, h),
            pos=(0 + x, self.height / 2 - (h / 2) + y),
            units='pix'
        )

        self._background = GratingStim(
            self,
            tex=np.full([16, 16], color),
            size=(self.width, self.height),
            units='pix'
        )

        self.getWarpMask(fill=color)

        self._background.draw()
        self.flip()

        return

    def getWarpMask(self, fill=0, clear=False):
        """
        """

        #
        array = np.full([self.height, self.width], fill).astype(np.float64)
        image = ImageStim(self, image=array, size=self.size, units='pix')
        image.draw()
        self.flip()
        warped = np.array(
            self.getMovieFrame(buffer='front').convert('L')
        )

        #
        if clear:
            self.flip()

        #
        self.mask = warped != 0

        return self.mask

    def wait(self, t=1, fill=0):
        """
        """

        array = np.full([self.height, self.width], fill).astype(np.float64)
        image = ImageStim(self, image=array, size=self.size, units='pix')
        for iframe in range(int(np.ceil(self.fps * t))):
            image.draw()
            self.flip()

        return

    def flip(self, draw_signal_patch=True, draw_background=False, **kwargs):
        """
        """

        if draw_background:
            self._background.draw()

        if draw_signal_patch:
            self.patch.draw()

        # Draw the signal patch
        if self.signalFrameCount != 0:
            self.patch.draw()
            self.signalFrameCount -= 1
        else:
            if self.state != False:
                self.state = False

        return super().flip(**kwargs)

    def flashSignalPatch(self, frameCount=3):
        """
        """

        self.state = True
        self.signalFrameCount = frameCount

        return

    def clearStimuli(self):
        """
        """

        self._background.draw()
        self.flip()

        return

    @property
    def warper(self):
        return self._warper

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
        fill = 1 if value is True else -1
        self.patch.tex = np.full([16, 16], fill)
        self._state = value

        return

    @property
    def patch(self):
        return self._patch

class DLPLightCrafter3010(Window):
    """
    """

    def __init__(
        self,
        ):
        """
        """

        self._width, self._height = size
        self._azimuth, self._elevation = fov
        self._fps = fps
        self._level = level

        super().__init__(
            size=(self.width, self.height),
            screen=screen,
            units='pix',
            gammaErrorPolicy='warn',
            useFBO=False,
        )

        #
        self._background = warping.SignaledAndWarpedStim(self)
        self.level = level

        return

    def clear(self):
        """
        Present the background stimulus
        """

        self.background.draw()
        self.flip()

        return

    def showBlankScreen(self, t=1, writer=None, draw_every_frame=False):
        """
        """

        for iframe in range(int(np.ceil(self.fps * t))):

            # Always draw the very first frame
            if iframe == 0:
                self.background.draw()
                self.flip()
                im = self.getMovieFrame(buffer='front', clear=False)

            # Draw other frames (optional)
            elif draw_every_frame:
                self.background.draw()
                self.flip()

            # Save to a video container (optional)
            if writer is not None:
                try:
                    writer.write(im)
                except:
                    continue

        return

    def getMovieFrame(self, buffer='back', clear=True):
        """
        """

        obj = super().getMovieFrame(buffer=buffer)
        image = np.array(obj.convert('L'))
        discard = self.movieFrames.pop()
        if clear:
            super().clearBuffer()

        return image

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
    def background(self):
        return self._background

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        """
        """

        if type(value) == str:
            if value not in ['black', 'gray', 'white', 'k', 'w']:
                raise Exception(f'Invalid value: {value}')
            elif value in ['black', 'k']:
                color = 0
            elif value in ['white', 'w']:
                color = 255
            else:
                color = 255 / 2

        elif type(value) in [int, float]:
            if value < 0 or value > 1:
                raise Exception(f'Invalid value: {value}')
            else:
                color = 255 * value

        array = np.full([self.height, self.width], color)
        self._background.array = array
        self._background.draw()
        self.flip()
        self._level = value

        return
