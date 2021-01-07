import cv2 as cv
import numpy as np


class PerspectiveTransform:
    """ Holds perspective transform matrices and applies transforms to an image. """
    def __init__(self, src, dst):
        self.M = cv.getPerspectiveTransform(src, dst)
        self.Minv = cv.getPerspectiveTransform(dst, src)

    def warp(self, img):
        """ Warps (from perspective to birds eye view) the image. """
        return cv.warpPerspective(img, self.M, img.shape[1::-1], flags=cv.INTER_LINEAR)

    def unwarp(self, img):
        """ Unwarps (from birds eye view to perspective) the image. """
        return cv.warpPerspective(img, self.Minv, img.shape[1::-1], flags=cv.INTER_LINEAR)
