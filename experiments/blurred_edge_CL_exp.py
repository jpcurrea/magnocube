"""Measure responses to a revolving 'edge' stimulus with different degrees of gaussian blur.

The edge stimulus is designed to have one maximum contrast edge followed by 
a continuous gradient. For instance, an ON edge goes from black to white and
then gradually darkens until it wraps around and becomes black at the edge 
again.

Parameters:
1) blur spread: 
2) edge polarity: ON or OFF edge
3) angular velocity: clockwise, counterclockwise, or still
4) rotational gain: 0 (open) or -1 (closed)

test duration of 10 s results in a 2 minute trial
"""
from datetime import datetime
import time
import holocube.hc as hc
import math
import numpy as np
from numpy import *
import os
from holocube.camera import TrackingTrial
from scipy import signal
import math
from matplotlib import pyplot as plt
from PIL import Image

def rgb2gray(arr):
    gray = .2989 * arr[..., 0] + .587 * arr[..., 1] + .1140 * arr[..., 2]
    return gray

def load_image(fn):
    img = Image.open(fn)
    arr = np.array(img)
    if arr.ndim > 2:
        arr = rgb2gray(arr)
    return arr

def set_attr_func(object, var, func):
    val = func()
    setattr(object, var, val)

def timestamp():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")

FOLDER = os.path.abspath("./blurred_edge_CL_exp")
# DURATION = int(5 * 60)    # seconds
DURATION = int(10)    # seconds
# DURATION = 15
# DURATION = 5   # seconds
SPEED = 135.0 * np.pi / 180.0
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)

# make background black
num_frames = DURATION * hc.scheduler.freq

# keep track of data as we run experiments
tracker = TrackingTrial(camera=hc.camera, window=hc.window, dirname=FOLDER)
tracker.add_virtual_object(name='bg', motion_gain=0,
                           start_angle=hc.camera.update_heading)

backgrounds = {}
bottom, top = -np.arctan2(1, 2*np.sqrt(2)), np.arctan2(3, 2*np.sqrt(2))

sequence_length = 2**8
signal = np.zeros(sequence_length, dtype='uint8')
vals = np.linspace(0, 255, sequence_length)
signal[:sequence_length//2] = vals[sequence_length//2:]
signal[sequence_length//2:] = vals[:sequence_length//2]
signal_off = signal
signal_off = np.repeat(signal_off[None], sequence_length, axis=0)
# make into a 4 channel 8 bit image
signal_off = np.stack([signal_off, signal_off, signal_off, 255 * np.ones_like(signal_off)], axis=-1)
signal_on = signal[::-1]
signal_on = np.repeat(signal_on[None], sequence_length, axis=0)
# make into a 4 channel 8 bit image
signal_on = np.stack([signal_on, signal_on, signal_on, 255 * np.ones_like(signal_on)], axis=-1)

edge_off = hc.stim.Quad_image(
    hc.window, left=0, right=2*np.pi, bottom=bottom, top=top, xres=sequence_length, 
    yres=sequence_length, xdivs=64, ydivs=1, dist=2)
edge_off.set_image(signal_off)
edge_on = hc.stim.Quad_image(
    hc.window, left=0, right=2*np.pi, bottom=bottom, top=top, xres=sequence_length, 
    yres=sequence_length, xdivs=64, ydivs=1, dist=2)
edge_on.set_image(signal_on)


# stds = np.append([0], 2**np.arange(4))
stds = [0, 1, 2, 4, 8, 16]

# define test parameters
exp_starts = [[hc.window.set_far, 3],
              [hc.window.set_bg, [0.0, 0.0, 0.0, 0.0]],  # 0-1
              [tracker.h5_setup],
              [hc.camera.clear_headings],
              [hc.camera.storing_start, -1, FOLDER, None, True],
              [tracker.store_camera_settings],
            #   [tracker.virtual_objects['bg'].set_motion_parameters, -1, hc.camera.update_heading],
              [tracker.add_exp_attr, 'video_fn', hc.camera.get_save_fn],
              [tracker.add_exp_attr, 'experiment', os.path.basename(FOLDER)],
              [tracker.add_exp_attr, 'start_exp', time.time],
              ]
exp_ends = [[hc.window.set_far,     1],
            [hc.window.set_bg, [.5, .5, .5, 1.0]],
            [tracker.add_exp_attr, 'stop_exp', time.time],
            [tracker.save],
            [hc.camera.storing_stop],
            ]
hc.scheduler.add_exp(name=os.path.basename(FOLDER), starts=exp_starts, ends=exp_ends)

# bar_oris += np.pi

for lbl, stim in zip(['edge_on', 'edge_off'], [edge_on, edge_off]):
    for std in stds:
        starts = [
            [stim.blur_image, std],
            [stim.switch, True],
            [hc.camera.import_config],
            [hc.camera.clear_headings],
            [tracker.virtual_objects['bg'].set_motion_parameters, -1, hc.camera.update_heading],
            # [stim.set_ry, hc.camera.update_heading],
            [hc.window.record_start],
            [print, f"stim: {lbl},\tvelo: {0},\tstd: {std}"]
        ]
        middles = [
            [hc.camera.get_background, hc.window.get_frame],
            [tracker.update_objects, hc.camera.update_heading],
            [stim.set_ry, tracker.virtual_objects['bg'].get_angle],
            [hc.window.record_frame]
        ]
        ends = [
            [stim.switch, False],
            [tracker.reset_virtual_object_motion],
            [tracker.add_test_data, hc.window.record_stop,
                {'stop_test': time.time, 'stim': lbl, 'velo': 0, 'std': std}, True],
            [hc.window.reset_rot],
        ]
        hc.scheduler.add_test(num_frames, starts, middles, ends)

# num_frames = 5 * hc.scheduler.freq
# pts = hc.stim.Points(hc.window, 1000, dims=[(-5, 5), (-5, 5), (-5, 5)], color=1, pt_size=4)
# headings = np.linspace(0, 720 * np.pi / 180., 480)
# headings = np.append(headings, headings[::-1])

# starts = [[pts.switch, True],
#           [hc.camera.reset_display_headings],
#           [tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, hc.camera.update_heading],
#           [tracker.virtual_objects['fly_heading'].add_motion, headings]
#          ]
# middles = [[hc.camera.import_config],
#            [hc.camera.get_background, hc.window.get_frame],
#            [tracker.update_objects, hc.camera.update_heading],
#            # [hc.window.set_rot, np.linspace(0, 2 * np.pi, 100)[:, None]],
#         #    [print, tracker.virtual_objects['fly_heading'].get_angle],
#            [pts.set_rot, tracker.virtual_objects['fly_heading'].get_rot],
#         #    [hc.window.set_yaw, tracker.virtual_objects['fly_heading'].get_angle],
#           ]
# ends = [[tracker.add_test_data, hc.window.record_stop,
#             {'stop_test': time.time,
#              'bg': 'points', 'bg_angle': tracker.virtual_objects['fly_heading'].get_angles}, False],
#         [tracker.reset_virtual_object_motion],
#         [pts.switch, False],
#         [hc.camera.reset_display_headings],
#         [hc.window.reset_rot]]
# hc.scheduler.add_rest(num_frames, starts, middles, ends)