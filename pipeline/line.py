from enum import Enum
from collections import deque

import numpy as np

from pipeline.sliding_window_config import SlidingWindowsConfig


class LinePosition(Enum):
    LEFT = 0
    RIGHT = 1


class Line():
    def __init__(self, pos: LinePosition, sliding_windows_config: SlidingWindowsConfig, stabilization_history_size: int = 10):
        self.detected = False
        self.fits = deque(maxlen=stabilization_history_size)
        self.a = self.b = self.c = 0
        self.pos = pos
        self.cfg = sliding_windows_config

    def detect(self, img, nonzero_x, nonzero_y):
        if not self.fits:
            self._detect_sliding_windows(img, nonzero_x, nonzero_y)
        else:
            self._detect_around_polynome(img, nonzero_x, nonzero_y)

    def curve(self, y):
        return self.a * (y**2) + self.b * y + self.c

    def _detect_sliding_windows(self, img, nonzero_x, nonzero_y):
        current_x = self._find_bottom_pos(img)
        window_height = np.int(img.shape[0] // self.cfg.number_of_windows)
        inds = []

        for window in range(self.cfg.number_of_windows):
            win_y_bottom = img.shape[0] - (window + 1) * window_height
            win_y_top = img.shape[0] - window * window_height
            win_x_bottom = current_x - self.cfg.window_margin
            win_x_top = current_x + self.cfg.window_margin

            good_y = (win_y_bottom <= nonzero_y) & (nonzero_y < win_y_top)
            good_inds = (good_y & (win_x_bottom <= nonzero_x)
                & (nonzero_x < win_x_top)).nonzero()[0]

            inds.append(good_inds)
            if len(good_inds) > self.cfg.min_pixels_for_detect:
                current_x = np.int(np.mean(nonzero_x[good_inds]))

        try:
            inds = np.concatenate(inds)
        except ValueError:
            print("Unable to concatenate indices after Sliding Window Algorithm")
            pass

        self._add_fit(nonzero_x[inds], nonzero_y[inds])

    def _find_bottom_pos(self, img):
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        return np.argmax(histogram[:midpoint]) if self.pos == LinePosition.LEFT else np.argmax(histogram[midpoint:]) + midpoint

    def _detect_around_polynome(self, img, nonzero_x, nonzero_y):
        curve = self.curve(nonzero_y)
        ids = ((nonzero_x > (curve - self.cfg.window_margin)) & (nonzero_x < (curve + self.cfg.window_margin)))
        self._add_fit(nonzero_x[ids], nonzero_y[ids])

    def _add_fit(self, x, y):
        self.fits.append(np.polyfit(y, x, 2))
        self.a, self.b, self.c = np.mean(self.fits, axis=0)
