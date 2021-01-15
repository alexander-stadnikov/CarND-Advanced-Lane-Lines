import glob
from typing import Tuple
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from pipeline import Camera, PerspectiveTransform
from pipeline import Lane, SlidingWindowsConfig, Detector, SobelConfig, ColorConfig
from pipeline import l_channel_from_rgb, lab_b_channel_from_rgb

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
            (offset, 0), (W - offset,0), (offset, H), (W - offset, H)
        ])
    )

mask = [
    (550, 450), (750, 450), (1200, 682), (100, 682)
]

SCALE = (
    3.7 / 370, # m/px by X axis
    30 / 720   # m/px by Y axis
)

def show_images(img_1, name_2, img_2, f_name=None, binary=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    fig.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(img_1)
    ax1.set_title('Original', fontsize=30)
    if binary:
        ax2.imshow(img_2, cmap='gray')
    else:
        ax2.imshow(img_2)
    ax2.set_title(name_2, fontsize=30)
    plt.tight_layout()
    if f_name is None:
        plt.show()
    else:
        plt.savefig(f_name, bbox_inches='tight')

def process_video(detector: Detector, video_file_name: str, range: Tuple[int, int] = None) -> None:
    if range is None:
        v_in = VideoFileClip(f"./{video_file_name}.mp4")
    else:
        v_in = VideoFileClip(f"./{video_file_name}.mp4").subclip(range[0], range[1])
    v_out = v_in.fl_image(detector.pipeline)
    v_out.write_videofile(f"./output_images/{video_file_name}_out.mp4", audio=False)

def basic_detector():
    return Detector(
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

def hls_channel(img, n):
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    channel = hls[:,:,n]
    return np.dstack((channel, channel, channel))

def lab_channel(img, n):
    lab = cv.cvtColor(img, cv.COLOR_RGB2Lab)
    channel = lab[:,:,n]
    return np.dstack((channel, channel, channel))

for f in glob.glob("./test_images/test2.jpg"):
    img = plt.imread(f)
    f_name = f"./output_images/{Path(f).stem}"

    transform = get_transform(464)
    warp_img = transform.warp(img)

    show_images(img, "Bird's-eye-view", warp_img, f"{f_name}_warp.png")

    l_channel = l_channel_from_rgb(warp_img, (200, 255))
    lab_b_channel = lab_b_channel_from_rgb(warp_img, (190, 255))
    poi = np.zeros_like(l_channel)
    poi[(l_channel == 1) | (lab_b_channel == 1)] = 1

    show_images(warp_img, 'HLS L-channel', hls_channel(warp_img, 1), f"{f_name}_hls_l.png")
    show_images(warp_img, 'HLS S-channel', hls_channel(warp_img, 2), f"{f_name}_hls_s.png")
    show_images(warp_img, 'LAB b-channel', lab_channel(warp_img, 2), f"{f_name}_lab_b.png")
    show_images(warp_img, 'Combined', poi, f"{f_name}_combined.png", True)

    img_processed = basic_detector().pipeline(img)
    show_images(img, 'Processed', img_processed, f"{f_name}.png")

process_video(basic_detector(), "project_video")

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
process_video(detector, "challenge_video")

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
