import numpy as np
import pathlib as pl
from scipy.ndimage import gaussian_filter as gfilt

def makeAlphaMask(shape=(720, 1280), margin=30, sigma=10, low=-1, high=1):
    """
    """

    mask = np.full(shape, high).astype(float)
    # mask = np.ones(shape)
    mask[:,  :margin] = low
    mask[:, -margin:] = low
    mask[:margin, :]  = low
    mask[-margin:, ]  = low
    filtered = gfilt(mask, sigma)
    clipped = np.clip(filtered, -1, 1)

    return clipped

def cycleSignalPatch(display, cycle=(1, 1), nCycles=1):
    """
    """

    for iCycle in range(nCycles):
        display.state = True
        for iFrame in range(int(round(display.fps * cycle[0]))):
            display.drawBackground()
            display.flip()

        display.state = False
        for iFrame in range(int(round(display.fps * cycle[1]))):
            display.drawBackground()
            display.flip()

    return

def generateMetadataFilename(parent, tag, extension='.pkl'):
    """
    """

    existing = list(parent.glob(f'{tag}*'))
    n = len(existing) + 1
    filename = parent.joinpath(f'{tag}{n}{extension}')
    return filename

def estimateFrameCount(t, fps=60, roundingMethod='nearest'):
    """
    Convert time (in seconds) to frames
    """

    if roundingMethod == 'nearest':
        nFrames = int(round(fps * t))
    elif roundingMethod == 'ceiling':
        nFrames = int(np.ceil(fps * t))
    elif roundingMethod == 'floor':
        nFrames = int(np.floor(fps * t))
    else:
        raise Exception(f'Rounding method must be one of "nearest", "ceiling", or "floor"')

    if nFrames == 0:
        raise Exception(f'Estimated frame count is 0')

    return nFrames