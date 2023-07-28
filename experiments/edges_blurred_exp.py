"""Test for the effect of an edge, blurred to see the effect of high spatial frequencies.

A 'single edge' stimulus is the attempt at generating a single high contrast edge with
an otherwise smooth transition in brightness. We measure flies' open- and then closed-loop responses
to this edge in different positions. 

Parameters:
1) position: cardinal directions and their midpoints (x8)
2) blur sigma: standard deviation of the gaussian blur (x6)
3) direction: the edge is either black then white or white then black (x2)
total= 8 * 6 * 2 tests = 96 tests

trial duration <= 300 s ==> test duration <= 3.12 s
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
from scipy import ndimage

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

FOLDER = os.path.abspath("./edges blurred")
DURATION = int(3)    # seconds
BLUR_VALS = 6
# DURATION = 15
# DURATION = 5   # seconds
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)

# make background black
num_frames = DURATION * hc.scheduler.freq

# keep track of data as we run experiments
hc.camera.update_heading()
tracker = TrackingTrial(camera=hc.camera, window=hc.window, dirname=FOLDER)
tracker.add_virtual_object(name='bg', motion_gain=-1,
                           start_angle=hc.camera.update_heading, object=True)

# load the images
fns = os.listdir("./experiments/natural_images")
fns = [os.path.join('experiments', 'natural_images', fn) for fn in fns]
fns = [os.path.abspath(fn) for fn in fns]
fns = [fn for fn in fns if fn.endswith('.jpg')]
imgs = [load_image(fn) for fn in fns]
res = 2**10
yres = res
# sigmas = np.append(0, np.logspace(1, 2, BLUR_VALS - 1))
sigmas = np.round(np.linspace(0, 200, BLUR_VALS)).astype(int)
backgrounds = []
img = np.zeros((yres, res, 4), dtype='uint8')
img[:] = np.linspace(0, 255, res, dtype='uint8')[np.newaxis, :, np.newaxis]
img[..., -1] = 255
for order in [1, -1]:
    cyl = hc.stim.Quad_image(hc.window, left=0, right=2*np.pi, bottom=-.2*pi,
                            top=.2*pi, xres=res, yres=yres, xdivs=64, ydivs=1)
    cyl.set_image(img[::order])
    backgrounds += [cyl]

# get the 8 key positions
orientations = np.arange(0, 8) * np.pi/4


# define test parameters
exp_starts = [[hc.window.set_far, 3],
              [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]],  # 0-1
              [tracker.h5_setup],
              [hc.camera.storing_start, -1, FOLDER, None, True],
              [tracker.store_camera_settings],
              [tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, hc.camera.update_heading],
              [tracker.virtual_objects['bg'].set_motion_parameters, 0, hc.camera.update_heading],
              [hc.camera.clear_headings],
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

bar_gain = 0
tracker.start_time = time.time()
# first, let's add the experiments with a background
for bg, fn in zip(backgrounds, fns):
    for orientation in orientations:
        for sigma in sigmas:
            starts = [
                [bg.switch, True],
                [bg.blur_image, sigma],
                [bg.set_ry, orientation],
                [hc.camera.import_config],
                [hc.camera.clear_headings],
                [hc.window.record_start],
                [print, f"fn={fn}"],
                [set_attr_func, tracker, 'start_time', time.time]
            ]
            middles = [
                [hc.camera.get_background, hc.window.get_frame],
                [tracker.update_objects, hc.camera.update_heading],
                [bg.set_ry, tracker.virtual_objects['bg'].get_angle],
                [hc.window.record_frame]
            ]
            ends = [
                [bg.switch, False],
                [tracker.reset_virtual_object_motion],
                [tracker.add_test_data, hc.window.record_stop,
                    {'img_fn': fn, 'start_test': getattr(tracker, 'start_time'), 'stop_test': time.time, 'sigma': sigma}, True],
                [hc.window.reset_rot],
            ]
            hc.scheduler.add_test(num_frames, starts, middles, ends)