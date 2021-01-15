""" The module contains detector related classes. """

from dataclasses import dataclass
from typing import Tuple, List
from collections import deque

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pipeline.tools import s_channel_from_rgb, h_channel_from_rgb, l_channel_from_rgb, lab_b_channel_from_rgb
from pipeline.transform import PerspectiveTransform
from pipeline.camera import Camera
from pipeline.lane import Lane
from pipeline import Sobel


@dataclass
class SobelConfig:
    direction_thrashold: Tuple[int, int]
    magnitude_trashold: Tuple[int, int]
    angle_trashold: Tuple[float, float]
    kernel_size: int


@dataclass
class ColorConfig:
    hls_l_trashold: Tuple[int, int]
    hls_s_trashold: Tuple[int, int]
    lab_b_trashold: Tuple[int, int]

class Detector:
    def __init__(self, camera: Camera, sobel_cfg: SobelConfig, color_cfg: ColorConfig,
                 transform: PerspectiveTransform, lane: Lane,
                 mask: List[Tuple[int, int]] = None, show_each_frame: bool = False):
        self.camera = camera
        self.sobel_cfg = sobel_cfg
        self.color_cfg = color_cfg
        self.transform = transform
        self.mask = None if mask is None else np.array([mask], dtype=np.int32)
        self.lane = lane
        self.show_each_frame = show_each_frame

    def pipeline(self, img):
        w, h = img.shape[1::-1]
        undistorted_img = self.camera.undistort(img)
        masked_img = self._apply_mask(undistorted_img)
        warped_img = self.transform.warp(masked_img)
        s_channel = s_channel_from_rgb(warped_img, self.color_cfg.hls_s_trashold)
        l_channel = l_channel_from_rgb(warped_img, self.color_cfg.hls_l_trashold)
        lab_b_channel = lab_b_channel_from_rgb(warped_img, self.color_cfg.lab_b_trashold)
        poi = np.zeros_like(s_channel)
        # poi[(s_channel == 1) | (l_channel == 1) | (lab_b_channel == 1)] = 1
        poi[(l_channel == 1) | (lab_b_channel == 1)] = 1

        if not self.sobel_cfg is None:
            sobel = self._sobel(undistorted_img)

        line_img = np.zeros((h, w, 3), dtype=np.uint8)
        lane_img, debug_img = self.lane.find_lane_lines(poi)
        poly_unwarp = self.transform.unwarp(lane_img)
        frame = cv.addWeighted(poly_unwarp, 0.8, undistorted_img, 1.0, 0.0)
        debug_img = cv.resize(debug_img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
        frame[:debug_img.shape[0], w//2:] = debug_img
        try:
            self._print(frame, (10, 50), f"Lane curvature: {int(np.abs(self.lane.curvature(h)))}(m)")
            self._print(frame, (10, 100), f"Car offset: {self.lane.car_offset((w, h)):.2f}(m)")
        except:
            pass

        if self.show_each_frame:
            self._show_plot('POI', poi, 1)
            self._show_plot('Result', frame)
        return frame

    def _sobel(self, img):
        x = (1, 0)
        y = (0, 1)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        sobel_src = cv.cvtColor(np.copy(img), cv.COLOR_RGB2HLS).astype(np.float)[:, :, 2]
        sobel_factory = Sobel(gray, kernel_size=self.sobel_cfg.kernel_size)\
            .by_direction(x, self.sobel_cfg.direction_thrashold)\
            .by_direction(y, self.sobel_cfg.direction_thrashold)\
            .by_magnitude(x, y, self.sobel_cfg.magnitude_trashold)\
            .by_angle(x, y, self.sobel_cfg.angle_trashold)
        return sobel_factory.build(
            ((sobel_factory[0] == 1) & (sobel_factory[1] == 1))
            | ((sobel_factory[2] == 1) & (sobel_factory[3] == 1)))

    def _apply_mask(self, poi):
        if self.mask is None:
            return poi

        mask = np.zeros_like(poi)
        if len(poi.shape) > 2:
            channel_count = poi.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv.fillPoly(mask, self.mask, ignore_mask_color)
        return cv.bitwise_and(poi, mask)

    def _print(self, frame, pos, msg):
        cv.putText(frame, msg, pos, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    def _show_plot(self, name, img, cmap=0):
        f, ax = plt.subplots(1, 1, figsize=(24, 7))
        if cmap==0:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(name, fontsize=50)
        f.tight_layout()
        plt.show()
