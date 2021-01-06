from __future__ import annotations
from functools import lru_cache

import cv2 as cv
import numpy as np

from .types import *


class Sobel:
    """ Implements the Sobel Operator """

    def __init__(self, img, kernel_size: int):
        self.binary = []
        self.id = 0
        self.img = img
        self.kernel_size = kernel_size

    def by_direction(self, direction: Direction, threshold: Threshold) -> Sobel:
        _min, _max = threshold
        sobel = self._sobel(direction)
        sobel_scaled = np.uint8(255 * sobel / np.max(sobel))
        self.binary.append(np.zeros_like(sobel_scaled))
        self.binary[-1][(sobel_scaled >= _min) & (sobel_scaled <= _max)] = 1
        return self

    def by_magnitude(self, x_direction: Direction, y_direction: Direction, threshold: Threshold):
        _min, _max = threshold
        magnitude = np.sqrt(self._sobel(x_direction) ** 2 + self._sobel(y_direction) ** 2)
        scale = np.max(magnitude) / 255
        magnitude = (magnitude / scale).astype(np.uint8)
        self.binary.append(np.zeros_like(magnitude))
        self.binary[-1][(magnitude >= _min) & (magnitude <= _max)] = 1
        return self

    def by_angle(self, x_direction: Direction, y_direction: Direction, threshold: Threshold):
        _min, _max = threshold
        grad_dir = np.arctan2(self._sobel(x_direction), self._sobel(y_direction))
        self.binary.append(np.zeros_like(grad_dir))
        self.binary[-1][(grad_dir >= _min) & (grad_dir <= _max)] = 1
        return self

    def build(self, expression):
        out = np.zeros_like(self.img)
        out[expression] = 1
        return out

    def __getitem__(self, item):
        return self.binary[item]

    @lru_cache
    def _sobel(self, direction: Tuple[int, int]) -> np.ndarray:
        return np.absolute(cv.Sobel(self.img, cv.CV_64F, direction[0], direction[1], self.kernel_size))
