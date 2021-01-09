from enum import Enum

import numpy as np

from pipeline.sliding_window_config import SlidingWindowsConfig


class LinePosition(Enum):
    LEFT = 0
    RIGHT = 1


class Line():
    def __init__(self, pos: LinePosition, sliding_windows_config: SlidingWindowsConfig):
        self.detected = False
        self.fit = None
        self.pos = pos
        self.cfg = sliding_windows_config

    def detect(self, img, nonzero_x, nonzero_y):
        if self.fit is None:
            self._detect_sliding_windows(img, nonzero_x, nonzero_y)
        else:
            self._detect_around_polynome(img, nonzero_x, nonzero_y)

    def curve(self, y):
        return self.fit[0] * (y**2) + self.fit[1] * y + self.fit[2]

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

        self.fit = np.polyfit(nonzero_y[inds], nonzero_x[inds], 2)

    def _find_bottom_pos(self, img):
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        return np.argmax(histogram[:midpoint]) if self.pos == LinePosition.LEFT else np.argmax(histogram[midpoint:]) + midpoint

    def _detect_around_polynome(self, img, nonzero_x, nonzero_y):
        curve = self.fit[0] * (nonzero_y**2) + self.fit[1] * nonzero_y + self.fit[2]
        ids = ((nonzero_x > (curve - self.cfg.window_margin)) & (nonzero_x < (curve + self.cfg.window_margin)))
        x = nonzero_x[ids]
        y = nonzero_y[ids]
        self.fit = np.polyfit(y, x, 2)
