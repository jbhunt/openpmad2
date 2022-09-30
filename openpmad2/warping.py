import os
import re
import pickle
import numpy as np
import pathlib as pl
from skimage import transform as tf
from skimage import exposure as exp
from psychopy.visual import ImageStim, GratingStim

WARPFILE  = None
TRANSFORM = None

def load_tform_data(display='DLPLightCrafter3010', date='2022-02-02', dst=None):
    """
    """

    print(f'Loading transformation for {display} on {date}')

    lut = None
    cwd = pl.Path(__file__)
    folder = cwd.parent.joinpath('data', 'tables')
    result = folder.rglob(f'*{display}*')
    for file in result:
        if date in str(file):
            path = pl.Path().joinpath(folder, file)
            with open(str(path), 'rb') as stream:
                try:
                    lut = pickle.load(stream)
                except:
                    return
            break

    if lut is None:
        raise Exception(f'No lookup table found for {display} on {date}')

    src = np.array(lut['src'])
    if dst is None:
        dst = np.array(lut['dst'])

    global TRANSFORM
    TRANSFORM = tf.PiecewiseAffineTransform()
    TRANSFORM.estimate(src, dst)

    return

load_tform_data()

def load_wfile_data(date='2022-08-25'):
    """
    Set the warpfile filepath global variable
    """

    wfile = None
    cwd = pl.Path(__file__)
    folder = cwd.parent.joinpath('data', 'warpfiles')
    result = folder.rglob(f'*warpfile*')
    for file in result:
        if date in str(file):
            path = pl.Path().joinpath(folder, file)
            wfile = str(path)

    if wfile is not None:
        global WARPFILE
        WARPFILE = wfile

    return

load_wfile_data()

def warp(image, rescale=False):
    """
    Warp any arbitrary image
    """

    global TRANSFORM
    if TRANSFORM is None:
        raise exception('Affine transformation has not been estimated (or estimation failed)')

    warped = tf.warp(
        image,
        TRANSFORM.inverse,
        output_shape=image.shape,
        preserve_range=True,
        cval=0,
        mode='constant'
    )

    return warped

class WarpedNumPyArrayStim():
    """
    Uses PsychoPy's ImageStim class to present warped NumPy arrays
    """

    def __init__(self, display, array=None):
        """
        """

        # Setup
        self._display = display
        self._array = array
        self._heart = ImageStim(display, size=display.size, units='pix')

        # Warp and rescale the array
        if array is not None:
            self.array = array

        return

    def draw(self):
        self.heart.draw()

    @property
    def array(self):
        return self._array

    @property
    def warped(self):
        return self._heart.image

    @array.setter
    def array(self, value):
        """
        Warp and rescale the target image/array
        """

        warped = warp(value).astype(np.float64)
        rescaled = exp.rescale_intensity(warped, in_range=(0, 255), out_range=(-1, 1))
        self._heart.image = rescaled
        self._array = value

        return

    @property
    def heart(self):
        return self._heart

    @property
    def display(self):
        return self._display

class SignaledAndWarpedStim(WarpedNumPyArrayStim):
    """
    Extension of the WarpedNumPyArrayStim class which lets you signal low or
    high states over a small subregion of the display
    """

    def __init__(
        self,
        display,
        state=False,
        subregion=(-15, 0, 30, 40),
        ):
        """
        """

        super().__init__(display)

        #
        unwarped = np.full([self.display.height, self.display.width], False).astype(bool)
        x, y, w, h = subregion
        row1, row2 = y, y + h
        col1, col2 = int(self.display.width / 2) + x, int(self.display.width / 2) + x + w
        unwarped[row1: row2, col1: col2] = True
        self._subregion = warp(unwarped.astype(float)).astype(bool)

        #
        unwarped = np.full([self.display.height, self.display.width], 1)
        self._region = warp(unwarped.astype(float)).astype(bool)
        self._region[self.subregion] = False

        #
        self._state = state

        return

    def draw(self):
        """
        """

        swap = np.copy(self._heart.image)

        if self.state:
            swap[self.subregion] =  1
        elif self.state is False:
            swap[self.subregion] = -1
        else:
            pass

        self._heart.image = swap

        super().draw()

        return

    @property
    def subregion(self):
        return self._subregion

    @property
    def region(self):
        return self._region

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if value in [0, -1, False, 'low']:
            self._state = False
        elif value in [1, True, 'high']:
            self._state = True
        elif value in ['none', 'None', None]:
            self._state = None
        else:
            raise Exception(f'Invalid value for state: {value}')

        return
