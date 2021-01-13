from typing import Tuple

import numpy as np
import cv2 as cv

from pipeline.line import LinePosition, Line
from pipeline.sliding_window_config import SlidingWindowsConfig


class Lane:
    """ Finds lane lines, their curvature, CCP and overlays if needed. """

    def __init__(self, sliding_windows_config: SlidingWindowsConfig, scale: Tuple[float, float], debug: bool = False):
        self.left_line = Line(LinePosition.LEFT, sliding_windows_config, debug=debug)
        self.right_line = Line(LinePosition.RIGHT, sliding_windows_config, debug=debug)
        self.scale = scale
        self.debug = debug
        self.bad_frames = 0
        self.max_bad_frames = 5
        self.reset_is_needed = True

    def _sanity_check(self, img):
        bottom = img.shape[0]
        ll = self.left_line
        rl = self.right_line
        left_curvature = ll.curvature_last(self.scale, bottom)
        right_curvature = rl.curvature_last(self.scale, bottom)
        curvatures_too_differ = (np.abs(left_curvature - right_curvature) > 2000 and min(left_curvature, right_curvature) < 5000)
        distance_too_big = (np.abs(ll.curve_last(bottom) - rl.curve_last(bottom))*self.scale[0] > 5)
        non_parallel = np.abs(ll.a - rl.a) > 0.001
        if ll.fits and rl.fits and (curvatures_too_differ or distance_too_big or non_parallel):
            self.bad_frames += 1
            if self.bad_frames == self.max_bad_frames:
                self.reset_is_needed = True
            return
        self.bad_frames = 0
        self.reset_is_needed = False
        self.left_line.store_last_detect()
        self.right_line.store_last_detect()


    def find_lane_lines(self, img):
        nonzero = img.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        if self.reset_is_needed:
            out_left_img = self.left_line._detect_sliding_windows(img, nonzerox, nonzeroy)
            out_right_img = self.right_line._detect_sliding_windows(img, nonzerox, nonzeroy)
        else:
            out_left_img = self.left_line._detect_around_polynome(img, nonzerox, nonzeroy)
            out_right_img = self.right_line._detect_around_polynome(img, nonzerox, nonzeroy)

        self._sanity_check(img)

        empty_img = np.zeros_like(img).astype(np.uint8)
        lane_img = np.dstack((empty_img, empty_img, empty_img))
        debug_img = np.dstack((empty_img, empty_img, empty_img))
        lane_poly_img = np.zeros_like(lane_img)

        y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_curve = self.left_line.curve(y)
        right_curve = self.right_line.curve(y)
        x_left = np.array([np.transpose(np.vstack([left_curve, y]))])
        x_right = np.array([np.flipud(np.transpose(np.vstack([right_curve, y])))])
        points = np.hstack((x_left, x_right))
        cv.fillPoly(lane_poly_img, np.int_([points]), (0, 255, 0))

        if self.debug:
            for d in range(len(nonzero[0])):
                y, x = nonzero[0][d], nonzero[1][d]
                cv.circle(debug_img, (x,y), radius=0, color=(255, 0, 0), thickness=10)

        lane_img = cv.addWeighted(lane_img, 1, lane_poly_img, 0.3, 0)

        if self.debug:
            debug_img = cv.addWeighted(debug_img, 1, out_left_img, 0.3, 0)
            debug_img = cv.addWeighted(debug_img, 1, out_right_img, 0.3, 0)

        return lane_img, debug_img

    def curvature(self, y: int) -> int:
        return np.average([
            self.left_line.curvature(self.scale, y),
            self.right_line.curvature(self.scale, y)
        ])


    def car_offset(self, frame_size: Tuple[int, int]) -> float:
        x, y = frame_size
        left_line_pos = self.left_line.curve(y)
        right_line_pos = self.right_line.curve(y)
        lane_center = np.abs(left_line_pos - right_line_pos)//2 + left_line_pos
        return self.scale[0]*np.abs(x//2 - lane_center)
