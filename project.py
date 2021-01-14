import glob
from typing import Tuple

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from pipeline import Camera, PerspectiveTransform
from pipeline import Lane, SlidingWindowsConfig, Detector, SobelConfig, ColorConfig


camera = Camera("./debug_output/calibration.cache")
if not camera.calibrated:
    camera.calibrate_with_chessboard(
        glob.glob("./camera_cal/*.jpg"),
        pattern_size=(9, 6),
        output="debug_output"
    )

def get_transform(y):
    offset = 450
    bottom = 682
    apex_y = 464
    Y = np.array([apex_y, bottom])
    a1, b1 = np.polyfit(Y, np.array([575, 258]), 1)
    a2, b2 = np.polyfit(Y, np.array([707, 1049]), 1)
    W, H = 1280, 720

    return PerspectiveTransform(
        src=np.float32([
            (a1*y + b1, y), (a2*y + b2, y), (258, bottom), (1049, bottom)
        ]),
        dst=np.float32([
            (offset, 0), (W - offset,0), (offset, H), (W - offset, H)])
    )

mask = [
    (550, 450), (750, 450), (1200, 682), (100, 682)
]

SCALE = (
    3.7 / 370, # m/px by X axis
    30 / 720   # m/px by Y axis
)

def process_video(detector: Detector, video_file_name: str, range: Tuple[int, int] = None) -> None:
    if range is None:
        v_in = VideoFileClip(f"./{video_file_name}.mp4")
    else:
        v_in = VideoFileClip(f"./{video_file_name}.mp4").subclip(range[0], range[1])
    v_out = v_in.fl_image(detector.pipeline)
    v_out.write_videofile(f"./{video_file_name}_out.mp4", audio=False)

detector = Detector(
    camera,
    None, # No Sobel
    ColorConfig(
        hls_l_trashold=(200, 255),
        hls_s_trashold=(80, 255),
        lab_b_trashold=(190, 255)
    ),
    get_transform(464),
    Lane(
        SlidingWindowsConfig(9, 100, 50),
        scale=SCALE,
        debug=True
    ),
    mask,
    show_each_frame=False
)
# process_video(detector, "project_video")

detector = Detector(
    camera,
    None, # No Sobel
    ColorConfig(
        hls_l_trashold=(200, 255),
        hls_s_trashold=(100, 255),
        lab_b_trashold=(190, 255)
    ),
    get_transform(500),
    Lane(
        SlidingWindowsConfig(9, 50, 50),
        scale=SCALE,
        debug=True
    ),
    mask,
    show_each_frame=False
)
# process_video(detector, "challenge_video")

detector = Detector(
    camera,
    None, # No Sobel
    ColorConfig(
        hls_l_trashold=(200, 255),
        hls_s_trashold=(200, 255),
        lab_b_trashold=(190, 255)
    ),
    get_transform(500),
    Lane(
        SlidingWindowsConfig(9, 50, 50),
        scale=SCALE,
        debug=True
    ),
    mask,
    show_each_frame=False
)
process_video(detector, "harder_challenge_video")

# for f in glob.glob("./test_images/*.jpg"):
#     img = plt.imread(f)
#     pipeline_frame(img)
