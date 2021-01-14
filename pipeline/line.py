from enum import Enum
from collections import deque
from typing import Tuple

import numpy as np
import cv2 as cv

from pipeline.sliding_window_config import SlidingWindowsConfig


class LinePosition(Enum):
    LEFT = 0
    RIGHT = 1


class Line():
    def __init__(self, pos: LinePosition, sliding_windows_config: SlidingWindowsConfig,
                 scale: Tuple[float, float], stabilization_history_size: int = 10):
        self.detected = False
        self.fits = deque(maxlen=stabilization_history_size)
        self.a = self.b = self.c = 0.0
        self.a_s = self.b_s = self.c_s = 0.0
        self.pos = pos
        self.cfg = sliding_windows_config
        self.scale = scale

    def reset(self):
        self.fits.clear()

    def store_last_detect(self):
        self.fits.append((self.a, self.b, self.c))

    def _mid_coefficients(self):
        return np.mean(self.fits, axis=0)

    def curve(self, y):
        a, b, c = self._mid_coefficients()
        return self._curve(a, b, c, y)

    def curve_last(self, y):
        return self._curve(self.a, self.b, self.c, y)

    def curvature(self, y: float) -> int:
        A, B = self.a_s, self.b_s
        _, ys = self.scale
        return (1 + (2*A*y*ys + B)**2)**1.5 / np.abs(2*A)

    def _curve(self, A, B, C, y):
        return A*(y**2) + B*y + C

    def detect_sliding_windows(self, img, nonzero_x, nonzero_y):
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        if self.pos == LinePosition.LEFT:
            current_x = np.argmax(histogram[:midpoint])
        else:
            current_x = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(img.shape[0] // self.cfg.number_of_windows)
        inds = []

        out_img = np.dstack((img, img, img))

        for window in range(self.cfg.number_of_windows):
            win_y_bottom = img.shape[0] - (window + 1) * window_height
            win_y_top = img.shape[0] - window * window_height
            win_x_bottom = current_x - self.cfg.window_margin
            win_x_top = current_x + self.cfg.window_margin

            cv.rectangle(out_img,(win_x_bottom,win_y_bottom),(win_x_top,win_y_top),(0, 255, 0), 2)

            good_y = (nonzero_y >= win_y_bottom) & (nonzero_y < win_y_top)
            good_inds = (good_y & (nonzero_x >= win_x_bottom) & (nonzero_x < win_x_top)).nonzero()[0]

            if len(good_inds) > self.cfg.min_pixels_for_detect:
                inds.append(good_inds)
                current_x = np.int(np.mean(nonzero_x[good_inds]))

        try:
            inds = np.concatenate(inds)
            self._add_fit(nonzero_x[inds], nonzero_y[inds])
        except ValueError:
            pass

        return out_img

    def detect_around_polynome(self, img, nonzero_x, nonzero_y):
        curve = self.curve(nonzero_y)
        ids = ((nonzero_x > (curve - self.cfg.window_margin)) & (nonzero_x < (curve + self.cfg.window_margin)))
        self._add_fit(nonzero_x[ids], nonzero_y[ids])

        out_img = np.dstack((img, img, img))
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        c = self.curve(ploty)
        left_window = np.array([np.transpose(np.vstack([c -self.cfg.window_margin, ploty]))])
        right_window = np.array([np.flipud(np.transpose(np.vstack([c+self.cfg.window_margin, ploty])))])
        points = np.hstack((left_window, right_window))
        cv.fillPoly(out_img, np.int_([points]), (0, 255, 0))

        return out_img

    def _add_fit(self, x, y):
        try:
            self.a, self.b, self.c = np.polyfit(y, x, 2)
            self.a_s, self.b_s, self.c_s = np.polyfit(y*self.scale[1], x*self.scale[0], 2)
        except:
            if self.fits:
                self.a, self.b, self.c = self.fits[-1]
            else:
                self.a, self.b, self.c = 0, 0, 0
