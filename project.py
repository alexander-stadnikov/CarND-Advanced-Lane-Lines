from pipeline import Camera
import glob
import cv2 as cv

c = Camera(glob.glob("./camera_cal/*.*"), calibration_pattern_size=(9, 6), debug_output=True)
img = cv.imread("./camera_cal/calibration1.jpg")
img = c.undistort(img)
cv.imshow('undistorted', img)
cv.waitKey(0)
cv.destroyAllWindows()
