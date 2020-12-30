import glob
import cv2 as cv
from pipeline import Camera

import matplotlib.pyplot as plt

camera = Camera("./debug_output/calibration.cache")
if not camera.calibrated:
    camera.calibrate_with_chessboard(glob.glob("./camera_cal/*.jpg"), pattern_size=(9, 6), output="debug_output")

img = cv.imread("./test_images/test4.jpg")
undistorted_img = camera.undistort(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted_img)
ax2.set_title('Undistorted Image', fontsize=50)
plt.show()