""" The module contains camera related classes and functions. """

import pickle
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Any


class Camera:
    """ The class implements the camera. """

    def __init__(self, calibration_images: List[str], calibration_pattern_size: Tuple[int, int],
                 debug_output: bool = False):
        self.mtx = None
        self.dist = None
        self.DEBUG_OUTPUT_DIR = "debug_output"
        self.CACHE_FILE_NAME = "camera.cache"

        if debug_output and not Path(self.DEBUG_OUTPUT_DIR).exists():
            Path(self.DEBUG_OUTPUT_DIR).mkdir()

        if Path(self.CACHE_FILE_NAME).exists():
            self._load()
        else:
            self._calibrate(calibration_images, calibration_pattern_size, debug_output)
            self._save()

    def undistort(self, img: Any):
        """ Returns undistorted image. """
        return cv.undistort(img, self.mtx, self.dist, None, self.mtx)

    def _calibrate(self, calibration_images: List[str], pattern_size: Tuple[int, int], debug_output: bool):
        w, h = pattern_size
        obj_points = []  # 3D real word points
        img_points = []  # 2D image points
        obj_coords = np.zeros((w * h, 3), np.float32)
        obj_coords[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        shape = None

        for file in calibration_images:
            img = cv.imread(file)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            res, corners = cv.findChessboardCorners(gray, pattern_size, None)
            shape = gray.shape[::-1]

            if not res:
                print(f"Unable to find chessboard corners on {file}")
                continue

            if debug_output:
                img = cv.drawChessboardCorners(img, pattern_size, corners, res)
                if not cv.imwrite(f"{self.DEBUG_OUTPUT_DIR}/{Path(file).name}", img):
                    raise IOError(f"Unable to store the debug output for {file}")

            img_points.append(corners)
            obj_points.append(obj_coords)

        _, self.mtx, self.dist, _, _ = cv.calibrateCamera(obj_points, img_points, shape, None, None)

    def _save(self):
        data = {
            "mtx": self.mtx,
            "dist": self.dist,
        }
        pickle.dump(data, open(self.CACHE_FILE_NAME, "wb"))

    def _load(self):
        data = pickle.load(open(self.CACHE_FILE_NAME, "rb"))
        self.mtx = data["mtx"]
        self.dist = data["dist"]
