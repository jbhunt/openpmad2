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
