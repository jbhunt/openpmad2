import numpy as np
from scipy.ndimage import gaussian_filter as gfilt

def make_alpha_mask(shape=(720, 1280), margin=30, sigma=10, low=-1, high=1):
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
