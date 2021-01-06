import cv2 as cv
import numpy as np

from .types import Threshold


def s_channel_from_rgb(img, threshold: Threshold):
    """ Extract thresholded S-channel from RGB image. """
    s_channel = cv.cvtColor(img, cv.COLOR_RGB2HLS)[:,:,2]
    out = np.zeros_like(s_channel)
    out[(s_channel > threshold[0]) & (s_channel <= threshold[1])] = 1
    return out
