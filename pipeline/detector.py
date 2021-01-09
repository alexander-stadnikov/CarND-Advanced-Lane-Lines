import numpy as np
import cv2 as cv

from pipeline.line import LinePosition, Line
from pipeline.sliding_window_config import SlidingWindowsConfig


class Detector:
    """ Finds lane lines, their curvature, CCP and overlays if needed. """

    def __init__(self, sliding_windows_config: SlidingWindowsConfig):
        self.left_line = Line(LinePosition.LEFT, sliding_windows_config)
        self.right_line = Line(LinePosition.RIGHT, sliding_windows_config)

    def find_lane_lines(self, img):
        nonzero = img.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        self.left_line.detect(img, nonzerox, nonzeroy)
        self.right_line.detect(img, nonzerox, nonzeroy)

        out_img = np.dstack((img, img, img)) * 255
        window_img = np.zeros_like(out_img)

        y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_curve = self.left_line.curve(y)
        right_curve = self.right_line.curve(y)
        x_left = np.array([np.transpose(np.vstack([left_curve, y]))])
        x_right = np.array([np.flipud(np.transpose(np.vstack([right_curve, y])))])
        points = np.hstack((x_left, x_right))
        cv.fillPoly(window_img, np.int_([points]), (0, 255, 0))

        return cv.addWeighted(out_img, 1, window_img, 0.3, 0)
