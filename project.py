import glob
import cv2 as cv
from pipeline import Camera

camera = Camera("./debug_output/calibration.cache")
if not camera.calibrated:
    camera.calibrate_with_chessboard(glob.glob("./camera_cal/*.jpg"), pattern_size=(9, 6), output="debug_output")

img = cv.imread("./camera_cal/calibration1.jpg")
undistorted_img = camera.undistort(img)
cv.imshow("Distorted", img)
cv.waitKey(0)
cv.imshow("Undistorted", undistorted_img)
cv.waitKey(0)
cv.destroyAllWindows()