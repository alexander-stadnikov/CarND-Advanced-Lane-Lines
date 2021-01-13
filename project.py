import glob
from typing import Tuple

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from pipeline import Camera, PerspectiveTransform
from pipeline import Lane, SlidingWindowsConfig, Detector, SobelConfig, HLSConfig


camera = Camera("./debug_output/calibration.cache")
if not camera.calibrated:
    camera.calibrate_with_chessboard(glob.glob("./camera_cal/*.jpg"), pattern_size=(9, 6), output="debug_output")

def create_detector_for_project_video(debug):
    offset = 200
    bottom = 720
    apex_y = 450
    apex_x = 1280//2
    apex_offset = 45
    dst_x_left = 300
    dst_x_right = 980
    mask_offset = 300
    mask = [
        (0, bottom), (apex_x - apex_offset - 100, apex_y - 50),
        (apex_x + apex_offset + 100, apex_y - 50), (1180, bottom)
    ]

    lane = Lane(
        SlidingWindowsConfig(9, 100, 50),
        scale=(3.7 / 515, 3 / 100), # meters per pixel in x and y dimension
        # scale=(3.7 / 700, 30 / 720), # meters per pixel in x and y dimension
        debug=True
    )

    transform = PerspectiveTransform(
        src=np.float32([
            [apex_x - apex_offset, apex_y], [apex_x + apex_offset, apex_y],
            [offset, bottom], [1280-offset, bottom]
        ]),
        dst=np.float32([
            [dst_x_left, 0], [dst_x_right, 0],
            [dst_x_left, bottom], [dst_x_right, bottom],
        ])
    )

    sobel_cfg = SobelConfig(
        direction_thrashold=(50, 100),
        magnitude_trashold=(50, 100),
        angle_trashold=(0.7, 1.3),
        kernel_size=3
    )

    hls_cfg = HLSConfig(s_trashold=(80, 255))
    return Detector(camera, sobel_cfg, hls_cfg, transform, lane, mask, show_each_frame=debug)

def create_detector_for_challenge_video(debug):
    offset = 200
    bottom = 720

    apex_y = 500
    apex_x = 1280//2
    apex_offset = 58

    dst_x_left = 300
    dst_x_right = 980
    mask_offset = 300

    mask = [
        (0, bottom), (apex_x - apex_offset - 100, apex_y - 50),
        (apex_x + apex_offset + 100, apex_y - 50), (1180, bottom)
    ]

    lane = Lane(
        SlidingWindowsConfig(9, 100, 50),
        scale=(3.7 / 515, 3 / 100), # meters per pixel in x and y dimension
        debug=True
    )

    transform = PerspectiveTransform(
        src=np.float32([
            [apex_x - apex_offset, apex_y], [apex_x + apex_offset, apex_y],
            [offset, bottom], [1280-offset, bottom]
        ]),
        dst=np.float32([
            [dst_x_left, 0], [dst_x_right, 0],
            [dst_x_left, bottom], [dst_x_right, bottom],
        ])
    )

    sobel_cfg = SobelConfig(
        direction_thrashold=(50, 100),
        magnitude_trashold=(50, 100),
        angle_trashold=(0.7, 1.3),
        kernel_size=3
    )

    hls_cfg = HLSConfig(s_trashold=(80, 255))
    return Detector(camera, sobel_cfg, hls_cfg, transform, lane, mask, show_each_frame=debug)

def process_video(detector: Detector, video_file_name: str, range: Tuple[int, int] = None) -> None:
    if range is None:
        v_in = VideoFileClip(f"./{video_file_name}.mp4")
    else:
        v_in = VideoFileClip(f"./{video_file_name}.mp4").subclip(range[0], range[1])
    v_out = v_in.fl_image(detector.pipeline)
    v_out.write_videofile(f"./{video_file_name}_out.mp4", audio=False)

detector = create_detector_for_project_video(False)
process_video(detector, "project_video")

# detector = create_detector_for_challenge_video()
# process_video(detector, "challenge_video", (4,5))

# for f in glob.glob("./test_images/*.jpg"):
#     img = plt.imread(f)
#     pipeline_frame(img)
