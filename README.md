# Advanced Lane Finding

This project is the continuation of the [Finding Lane Lines](https://github.com/alexander-stadnikov/CarND-Finding-Lane-Lines). This time the goal is to identify curved lane lines. I’ll recognize lines based on the perspective transform to a bird’s eye view of an original frame from the camera.

## The Project Overview
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.

* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## The Project Layout
---

The project is organized as a Python application. The [pipeline](https://github.com/alexander-stadnikov/CarND-Advanced-Lane-Lines/tree/main/pipeline) was implemented as a Python package.
The application uses the pipeline to perform all computations and is located in the file [project.py](https://github.com/alexander-stadnikov/CarND-Advanced-Lane-Lines/blob/main/project.py).

## Camera Calibration and Image Undistortion
---

The Camera Calibration is needed to avoid the [Optic Distortion](https://en.wikipedia.org/wiki/Distortion_(optics)) effect. Each camera has it's own distortion coefficients. These coefficients might be computed with a known geometric pattern. This project contains pictures of the chessboard patterns with high contrast and known geometry. The pattern's size is the number of inner corners – there're nine in each row and six corners in each column.

OpenCV has a very useful function [findChessboardCorners](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a). The function takes a grayscaled image, the geometry as arguments, and returns coordinates of found corners:

```python
img = cv.imread("calibration.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
res, corners = cv.findChessboardCorners(gray, (9, 6), None)
```

To compute the calibration coefficient and the internal camera matrix OpenCV offers [calibrateCamera](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d) function. It takes as arguments expected 3D object points, captured 2D image points, and the image's shape. To compute object points Z coordinate is assumed to be equal to zero. 

For more precise calibration, it’s recommended to use as many pictures as possible. For each image, it’s necessary to find chessboard corners and store them as a collection. Object points for each picture will be the same, and also organized a collection:

```python
import cv2 as cv
import numpy as np

w, h = 9, 6
obj_coords = np.zeros((w * h, 3), np.float32)
obj_coords[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
gray = None

for file in images:
    img = cv.imread(file)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    res, corners = cv.findChessboardCorners(gray, pattern_size, None)

    if not res:
        print(f"Unable to find chessboard corners on {file}")
        continue

    img_points.append(corners)
    obj_points.append(obj_coords)

shape = gray.shape[::-1]
_, mtx, dist, _, _ = cv.calibrateCamera(obj_points, img_points, shape, None, None)
```

Finally, an image might be [undistorted](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga69f2545a8b62a6b0fc2ee060dc30559d):

```python
img = cv.imread("image.jpg")
undistored_img = cv.undistort(img, mtx, dist, None, mtx)
```

Example:

![Example](output_images/undist_chessboard.png)

The full source code for the step is in the file [camera.py](https://github.com/alexander-stadnikov/CarND-Advanced-Lane-Lines/blob/main/pipeline/camera.py).