from dataclasses import dataclass
from typing import Tuple, List
from collections import deque

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pipeline.tools import s_channel_from_rgb, h_channel_from_rgb
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
class HLSConfig:
    s_trashold: Tuple[int, int]


class Detector:
    def __init__(self, camera: Camera, sobel_cfg: SobelConfig, hls_cfg: HLSConfig,
                 transform: PerspectiveTransform, lane: Lane,
                 mask: List[Tuple[int, int]] = None, show_each_frame: bool = False):
        self.camera = camera
        self.sobel_cfg = sobel_cfg
        self.hls_cfg = hls_cfg
        self.transform = transform
        self.mask = None if mask is None else np.array([mask], dtype=np.int32)
        self.lane = lane
        self.show_each_frame = show_each_frame
        self.curvature = deque(maxlen=10)
        self.offset = deque(maxlen=10)

    def pipeline(self, img):
        w, h = img.shape[1::-1]
        undistorted_img = self.camera.undistort(img)
        sobel = self._sobel(undistorted_img)
        # s_channel = s_channel_from_rgb(undistorted_img, (200, 255))
        s_channel = s_channel_from_rgb(undistorted_img, self.hls_cfg.s_trashold)
        # h_channel = h_channel_from_rgb(undistorted_img, (0, 30))

        # hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
        # self._show_plot(hls[:,:,0])
        # self._show_plot(hls[:,:,2])
        # self._show_plot(h_channel)
        # self._show_plot(s_channel)

        # self._show_plot(hls[:,:,2])
        # self._show_plot(img)
        poi = self._apply_mask(self._poi(sobel, s_channel))
        # self._show_plot(self.transform.warp(poi), 1)
        # self._show_plot(poi, 1)
        line_img = np.zeros((h, w, 3), dtype=np.uint8)
        lane_img, debug_img = self.lane.find_lane_lines(self.transform.warp(poi))
        poly_unwarp = self.transform.unwarp(lane_img)
        frame = cv.addWeighted(poly_unwarp, 0.8, undistorted_img, 1.0, 0.0)
        debug_img = cv.resize(debug_img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
        frame[:debug_img.shape[0], w//2:] = debug_img
        self.curvature.append(np.abs(self.lane.curvature(h)))
        self.offset.append(self.lane.car_offset((w, h)))
        self._print(frame, (10, 50), f"Lane curvature: {int(np.mean(self.curvature))}(m)")
        self._print(frame, (10, 100), f"Car offset: {np.mean(self.offset):.2f}(m)")
        if self.show_each_frame:
            self._show_plot(self.transform.warp(poi), 1)
            self._show_plot(frame)
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

    def _poi(self, sobel, s_channel):
        poi = np.zeros_like(s_channel)
        # poi[((s_channel == 1) & (h_channel != 1)) | (sobel == 1)] = 1
        # poi[(h_channel == 1) & (s_channel != 1) & (sobel != 1)] = 1
        # poi[(sobel == 1)] = 1
        # poi[(h_channel == 1)] = 1
        poi[(s_channel == 1) | (sobel == 1)] = 1
        return poi

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

    def _show_plot(self, img, cmap=0):
        f, ax = plt.subplots(1, 1, figsize=(24, 7))
        if cmap==0:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title('Original Image', fontsize=50)
        f.tight_layout()
        plt.show()
