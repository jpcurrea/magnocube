"""Test for the effect of revolving a motion defined bar on a moving background.

We will wiggle a (motion- or contrast-defined) bar back and forth for 10 seconds on each hemisphere. We predict that the 
motion-defined bar will ellicit more saccades than the contrast defined one. We'll also try to measure a visual region that
triggers more saccades by looking at the saccade histogram.

Parameters:
1) background: closed-loop present vs. absent (x2) 
2) bar type: dark, light, or motion-defined (x3)
3) hemisphere: left vs right (x2)
total= 2 * 3 * 2 tests = 12 tests

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

FOLDER = os.path.abspath("./natural images")
# DURATION = int(5 * 60)    # seconds
DURATION = int(5./4. * 60)    # seconds
# DURATION = 15
# DURATION = 5   # seconds
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)

# make background black
num_frames = DURATION * hc.scheduler.freq

# keep track of data as we run experiments
hc.camera.update_heading()
tracker = TrackingTrial(camera=hc.camera, window=hc.window, dirname=FOLDER)
tracker.add_virtual_object(name='bg', motion_gain=0,
                           start_angle=hc.camera.update_heading, object=True)

# load the images
# fns = os.listdir("./experiments/natural_images")
# fns = [os.path.join('experiments', 'natural_images', fn) for fn in fns]
fns = os.listdir("./experiments/calibrations")
fns = [os.path.join('experiments', 'calibrations', fn) for fn in fns]
fns = [os.path.abspath(fn) for fn in fns]
# fns = [fn for fn in fns if fn.endswith('.jpg')]
fns = [fn for fn in fns if fn.endswith('.png')]
imgs = [load_image(fn) for fn in fns]
backgrounds = []
for img in imgs:
    # upsample to the nearest power of 2
    order = math.ceil(np.log2(img.shape[1]))
    new_width = 2**order
    img = signal.resample(img, new_width, axis=1)
    # make into RGBA array
    height, width = img.shape
    new_img = np.repeat(img[..., np.newaxis], 4, axis=-1)
    new_img[..., 3] = 255
    # flip the image
    new_img[:] = new_img[::-1]
    cyl = hc.stim.Quad_image(hc.window, left= 0, right=2 * pi, bottom=-.2*pi,
                             top=.2*pi, xres=height,
                             yres=width, xdivs=64, ydivs=1)
    cyl.set_image(new_img.astype('uint8'))
    backgrounds += [cyl]

# make an array of orientations so that there is a 90 degree rotation
orientations = np.zeros(num_frames)
orientations[-num_frames//2:] = np.pi
# todo: make a contrast ramp down and up at the same time as the orientation change
ts = np.arange(num_frames) / (60 * hc.scheduler.freq)
# ts = np.arange(num_frames) / (hc.scheduler.freq)
contrasts = np.ones(num_frames)
# add a linear ramp down starting at 2 seconds and ending at 2.45 s
# ramp_down_ts = (ts > 2) * (ts <= 2.45)
ramp_down_ts = (ts > 2./4.) * (ts <= 2.45/4.)
contrasts[ramp_down_ts] = np.linspace(1, 0, sum(ramp_down_ts))
# set to 0 for .1 s
pause_ts = (ts > 2.45/4.) * (ts <= 2.55/4.)
contrasts[pause_ts] = 0
# ramp up starting at 2.55 and ending at 3 seconds
ramp_up_ts = (ts > 2.55/4.) * (ts <= 3./4.)
contrasts[ramp_up_ts] = np.linspace(0, 1, sum(ramp_up_ts))

# define test parameters
exp_starts = [[hc.window.set_far, 3],
              [hc.window.set_bg, [0., 0., 0., 0.]],  # 0-1
              [tracker.h5_setup],
              [hc.camera.storing_start, -1, FOLDER, None, True],
              [tracker.store_camera_settings],
              [tracker.virtual_objects['fly_heading'].set_motion_parameters, 0, hc.camera.update_heading],
            #   [tracker.virtual_objects['bg'].set_motion_parameters, -1, hc.camera.update_heading],
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
    starts = [
        [bg.switch, True],
        [tracker.virtual_objects['bg'].add_motion, orientations],
        [hc.camera.import_config],
        [hc.camera.clear_headings],
        [hc.window.record_start],
        [print, f"fn={fn}"],
        [set_attr_func, tracker, 'start_time', time.time]
    ]
    middles = [
        [hc.camera.get_background, hc.window.get_frame],
        [tracker.update_objects, hc.camera.update_heading],
        [bg.set_contrast, contrasts],
        [bg.set_ry, tracker.virtual_objects['bg'].get_angle],
        # [bar.set_ry, tracker.virtual_objects['bar'].get_angle],
        [hc.window.record_frame]
    ]
    ends = [
        [bg.switch, False],
        [tracker.reset_virtual_object_motion],
        [tracker.add_test_data, hc.window.record_stop,
            {'img_fn': fn, 'start_test': getattr(tracker, 'start_time'), 'stop_test': time.time}, True],
        [hc.window.reset_rot],
    ]
    hc.scheduler.add_test(num_frames, starts, middles, ends)