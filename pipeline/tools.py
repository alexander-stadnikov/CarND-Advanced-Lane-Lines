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
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    hls_s = hls[:,:,2]
    out = np.zeros_like(hls_s)
    out[(hls_s > threshold[0]) & (hls_s <= threshold[1])] = 1
    return out

def h_channel_from_rgb(img, threshold: Threshold):
    """ Extract thresholded H-channel from RGB image. """
    return n_channel_from_rgb(img, threshold, 0)

def l_channel_from_rgb(img, threshold: Threshold):
    """ Extract thresholded H-channel from RGB image. """
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    out = np.zeros_like(hls_l)
    out[(hls_l > threshold[0]) & (hls_l <= threshold[1])] = 1
    return out

def lab_b_channel_from_rgb(img, threshold: Threshold):
    """ Extract thresholded H-channel from RGB image. """
    lab = cv.cvtColor(img, cv.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    out = np.zeros_like(lab_b)
    out[((lab_b > threshold[0]) & (lab_b <= threshold[1]))] = 1
    return out
