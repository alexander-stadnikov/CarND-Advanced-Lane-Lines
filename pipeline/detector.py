from typing import Tuple

import numpy as np
import cv2 as cv

from pipeline.line import LinePosition, Line
from pipeline.sliding_window_config import SlidingWindowsConfig


class Detector:
    """ Finds lane lines, their curvature, CCP and overlays if needed. """

    def __init__(self, sliding_windows_config: SlidingWindowsConfig, scale: Tuple[float, float], debug: bool = False):
        self.left_line = Line(LinePosition.LEFT, sliding_windows_config, debug=debug)
        self.right_line = Line(LinePosition.RIGHT, sliding_windows_config, debug=debug)
        self.scale = scale
        self.debug = debug

    def find_lane_lines(self, img):
        nonzero = img.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        out_left_img = self.left_line.detect(img, nonzerox, nonzeroy)
        out_right_img = self.right_line.detect(img, nonzerox, nonzeroy)

        empty_img = np.zeros_like(img).astype(np.uint8)
        out_img = np.dstack((empty_img, empty_img, empty_img))
        window_img = np.zeros_like(out_img)

        y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_curve = self.left_line.curve(y)
        right_curve = self.right_line.curve(y)
        x_left = np.array([np.transpose(np.vstack([left_curve, y]))])
        x_right = np.array([np.flipud(np.transpose(np.vstack([right_curve, y])))])
        points = np.hstack((x_left, x_right))
        cv.fillPoly(window_img, np.int_([points]), (0, 255, 0))

        if self.debug:
            for d in range(len(nonzero[0])):
                y, x = nonzero[0][d], nonzero[1][d]
                cv.circle(out_img, (x,y), radius=0, color=(255, 0, 0), thickness=10)

        out_img = cv.addWeighted(out_img, 1, window_img, 0.3, 0)

        if self.debug:
            out_img = cv.addWeighted(out_img, 1, out_left_img, 0.3, 0)
            out_img = cv.addWeighted(out_img, 1, out_right_img, 0.3, 0)

        return out_img

    def curvature(self, y: int) -> int:
        return (self.left_line.curvature(self.scale, y) + self.right_line.curvature(self.scale, y)) // 2
