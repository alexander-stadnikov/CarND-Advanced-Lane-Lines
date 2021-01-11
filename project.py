import glob
import cv2 as cv
import numpy as np

from pipeline import Camera, Sobel, PerspectiveTransform
from pipeline import Detector, SlidingWindowsConfig
from pipeline.tools import h_channel_from_rgb, s_channel_from_rgb

import matplotlib.pyplot as plt

camera = Camera("./debug_output/calibration.cache")
if not camera.calibrated:
    camera.calibrate_with_chessboard(glob.glob("./camera_cal/*.jpg"), pattern_size=(9, 6), output="debug_output")

def show(img):
    cv.imshow('test', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

img = cv.imread("./test_images/straight_lines1.jpg")
undistorted_img = camera.undistort(img)
#show(undistorted_img)
gray = cv.cvtColor(undistorted_img, cv.COLOR_BGR2GRAY)
#show(gray)

#out_img = sobel_by_magnitude(gray, (1, 0))
#show(out_img)

def showp(ax, img, name):
    ax.imshow(img, cmap='gray')
    ax.set_title(name, fontsize=50)

# dir_thrash = (50, 200)
# k = 3

# sobel_x = sobel_by_direction(gray, (1, 0), dir_thrash, k)
# sobel_y = sobel_by_direction(gray, (0, 1), dir_thrash, k)
# sobel_mag = sobel_by_magnitude(gray, (100, 200), k)
# sobel_ang = sobel_by_angle(gray, (.7, 1.3), k)
# sobel_total = np.zeros_like(gray)
# sobel_total[((sobel_x == 1) & (sobel_y == 1)) | ((sobel_mag == 1) & (sobel_ang == 1))] = 1

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_dots(img, dots, color=[255, 0, 0], thickness=-1):
    nonz = dots.nonzero()
    for d in range(len(nonz[0])):
        y, x = nonz[0][d], nonz[1][d]
        cv.circle(img, (x,y), radius=0, color=(255, 0, 0), thickness=10)


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv.addWeighted(initial_img, α, img, β, γ)

def get_mask(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv.fillPoly(mask, vertices, ignore_mask_color)
    return mask


def remove_shadows(img):
    # imgc = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    rgb_planes = cv.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 21)
        diff_img = plane - cv.absdiff(plane, bg_img)
        norm_img = cv.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

        _, thr_img = cv.threshold(norm_img, 230, 0, cv.THRESH_TRUNC)
        norm_img = cv.normalize(thr_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv.merge(result_planes)
    result_norm = cv.merge(result_norm_planes)

    # show(result)
    # show(result_norm)

    return result_norm


offset = 200
bottom = 720
right = 1280
apex_y = 450

# apex_y = 600
apex_x = 1280//2
# apex_offset = 350
apex_offset = 58
# src = np.float32([[apex_x - apex_offset, apex_y], [apex_x + apex_offset, apex_y],
#     [offset, bottom], [1280-offset, bottom]])
src = np.float32([[apex_x - apex_offset, apex_y], [apex_x + apex_offset, apex_y],
    [100, bottom], [1180, bottom]])

dst_x_left = 300
dst_x_right = 980
dst = np.float32([
    [dst_x_left, 0], [dst_x_right, 0],
    [dst_x_left, bottom], [dst_x_right, bottom],
])
transform = PerspectiveTransform(src,dst)
mask = None
detector = Detector(
    SlidingWindowsConfig(9, 100, 50),
    scale=(3.7 / 515, # meters per pixel in x dimension
           3 / 100   # meters per pixel in y dimension
    ),
    debug=True
)

def show_plot(img, cmap=0):


    f, ax = plt.subplots(1, 1, figsize=(24, 7))
    if cmap==0:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap='gray')
    ax.set_title('Original Image', fontsize=50)
    f.tight_layout()
    plt.show()

def pipeline_frame(img):

    # detector = Detector(
    #     SlidingWindowsConfig(9, 100, 50),
    #     scale=(3.7 / 515, # meters per pixel in x dimension
    #         3 / 100   # meters per pixel in y dimension
    #     ),
    #     debug=True
    # )
    global transform, mask, apex_y, apex_x, apex_offset
    # print(img.shape)
    direction_thrashold = (50, 150)
    x = (1, 0)
    y = (0, 1)

    # undistorted_img = transform.warp(camera.undistort(img))
    undistorted_img = camera.undistort(img)

    gray = cv.cvtColor(undistorted_img, cv.COLOR_RGB2GRAY)

    s = Sobel(gray, kernel_size=3)\
        .by_direction(x, direction_thrashold)\
        .by_direction(y, direction_thrashold)\
        .by_magnitude(x, y, (100, 200))\
        .by_angle(x, y, (0.7, 1.3))
    sobel = s.build(((s[0] == 1) & (s[1] == 1)) | ((s[2] == 1) & (s[3] == 1)))
    # sobel = s.build(((s[3] == 1)))

    s_channel = s_channel_from_rgb(undistorted_img, (80, 255))
    h_channel = h_channel_from_rgb(undistorted_img, (90, 255))

    poi = np.zeros_like(s_channel)
    poi[((s_channel == 1) & (h_channel != 1)) | (sobel == 1)] = 1
    # poi[(s_channel == 1)] = 1
    # poi[(sobel == 1)] = 1
    # poi[(h_channel == 1)] = 1

    if mask is None:
        mask = get_mask(poi, np.array([
            [(0, 720), (apex_x - apex_offset, apex_y), (apex_x + apex_offset, apex_y), (1280, 720)]
        ], dtype=np.int32))

    # poi = cv.bitwise_and(poi, mask)

    line_img = np.zeros((undistorted_img.shape[0], undistorted_img.shape[1], 3), dtype=np.uint8)
    # draw_dots(line_img, sobel)
    poly = detector.find_lane_lines(transform.warp(poi))
    poly_unwarp = transform.unwarp(poly)
    # draw_dots(line_img, poly_unwarp)
    frame = weighted_img(undistorted_img, poly_unwarp)
    poly = cv.resize(poly, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    frame[:poly.shape[0], img.shape[1]//2:] = poly

    font                   = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    try:
        cv.putText(frame, f"Lane curvature: {int(np.abs(detector.curvature(720)))}(m)", bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    except:
        cv.putText(frame, f"Lane curvature: NaN", bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv.putText(frame, f"Car offset: {detector.car_offset((1280, 720)):.2f}(m)", (10, 100), font, fontScale, fontColor, lineType)
    # show_plot(result)
    # result = weighted_img(undistorted_img, line_img)


    # show_plot(poi)


    # show_plot(poly)
    # show_plot(frame)
    # f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 7))
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=50)
    # showp(ax2, sobel_x, 'dir X')
    # showp(ax3, sobel_y, 'dir Y')
    # showp(ax4, sobel_mag, 'mag')
    # showp(ax5, sobel_ang, 'ang')
    # showp(ax6, sobel_total, 'total')
    # f.tight_layout()
    # plt.show()

    return frame



from moviepy.editor import VideoFileClip

v_name = "project_video"

# for f in glob.glob("./test_images/*.jpg"):
#     img = plt.imread(f)
#     pipeline_frame(img)

# clip1 = VideoFileClip(f"./{v_name}.mp4").subclip(5, 16)
clip1 = VideoFileClip(f"./{v_name}.mp4")
white_clip = clip1.fl_image(pipeline_frame)
white_clip.write_videofile(f"./{v_name}_out.mp4", audio=False)
