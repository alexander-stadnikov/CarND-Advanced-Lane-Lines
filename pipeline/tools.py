import cv2 as cv
import numpy as np

from pipeline.types import Threshold


__all__ = [
    "s_channel_from_rgb",
    "h_channel_from_rgb",
]

def n_channel_from_rgb(img, threshold: Threshold, channel: int):
    """ Extract thresholded S-channel from RGB image. """
    channel = cv.cvtColor(img, cv.COLOR_RGB2HLS)[:, :, channel]
    out = np.zeros_like(channel)
    out[(channel > threshold[0]) & (channel <= threshold[1])] = 1
    return out

def s_channel_from_rgb(img, threshold: Threshold):
    """ Extract thresholded S-channel from RGB image. """
    return n_channel_from_rgb(img, threshold, 2)

def h_channel_from_rgb(img, threshold: Threshold):
    """ Extract thresholded H-channel from RGB image. """
    return n_channel_from_rgb(img, threshold, 0)
