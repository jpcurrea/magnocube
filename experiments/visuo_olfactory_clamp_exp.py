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

FOLDER = os.path.abspath("./olfactory_clamp_exp")
# DURATION = int(5 * 60)    # seconds
DURATION = int(5)    # seconds
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
                           start_angle=hc.camera.update_heading)

# load the images
# fns = os.listdir("./experiments/natural_images")
# fns = [os.path.join('experiments', 'natural_images', fn) for fn in fns]
fns = os.listdir("./experiments/natural_images")
fns = [os.path.join('experiments', 'natural_images', fn) for fn in fns]
fns = [os.path.abspath(fn) for fn in fns]
# fns = [fn for fn in fns if fn.endswith('.jpg')]
fns = [fn for fn in fns if fn.endswith('.jpg')]
imgs = [load_image(fn) for fn in fns]
backgrounds = {}
bottom, top = -np.arctan2(1, 2*np.sqrt(2)), np.arctan2(3, 2*np.sqrt(2))
for img in imgs[:1]:
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
    cyl = hc.stim.Quad_image(hc.window, left= 0, right=2 * pi, bottom=bottom,
                             top=top, xres=height,
                             yres=width, xdivs=64, ydivs=1)
    cyl.set_image(new_img.astype('uint8'))
    backgrounds['natural'] = cyl

# make some simple visual stimuli:
# a single 30 degree dark bar
mean_val = new_img[..., 0].mean()
new_img = np.zeros((height, width, 4), dtype='uint8')
new_img[..., -1] = 255
# get the angle for each pixel
angles = np.linspace(0, 2 * np.pi, width)
in_bar = (angles < np.pi / 12) + (angles > 2*np.pi - np.pi / 12)
new_img[:, in_bar, :3] = 0
# fill with appropriate gray to keep the mean luminance the same as the natural image
# let's keep the total luminance the same as the natural image
# this means that N * mean_val = 0 * (in_bar) + other_val * (not in_bar)
# => other_val = N * mean_val / (not in_bar)
other_val = width * mean_val / (~in_bar).sum()
new_img[:, in_bar == False, :3] = other_val.astype('uint8')
bar = hc.stim.Quad_image(hc.window, left=0*pi, right=2*pi,
                         bottom=bottom, top=top, xres=height, yres=width, xdivs=64, ydivs=1,
                         dist=1)
bar.set_image(new_img)
backgrounds['bar'] = bar
# a repeating 30 degree dark bar, making a 30 degree wavelength grating
new_img = np.zeros((height, width, 4), dtype='uint8')
in_grating = (angles % (2 * np.pi / 6)) < np.pi / 6
new_img[..., -1] = 255
new_img[:, in_grating, :3] = 0
other_val = width * mean_val / (~in_grating).sum()
new_img[:, in_grating == False, :3] = other_val.astype('uint8')
grating = hc.stim.Quad_image(hc.window, left=0*pi, right=2*pi,
                             bottom=bottom, top=top, xres=height, yres=width, xdivs=64, ydivs=1,
                             dist=1)
grating.set_image(new_img)
backgrounds['grating'] = grating
# now make a uniform gray background that is treated like the others
new_img[..., :3] = 128
uniform = hc.stim.Quad_image(hc.window, left=0*pi, right=2*pi,
                             bottom=bottom, top=top, xres=height, yres=width, xdivs=64, ydivs=1,
                             dist=1)
uniform.set_image(new_img)
backgrounds['uniform'] = uniform



# define test parameters
exp_starts = [[hc.window.set_far, 3],
              [hc.window.set_bg, [0.5, 0.5, 0.5, 1]],  # 0-1
              [tracker.h5_setup],
              [hc.camera.storing_start, -1, FOLDER, None, True],
              [tracker.store_camera_settings],
              [tracker.virtual_objects['fly_heading'].set_motion_parameters, 0, hc.camera.update_heading],
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
# add multiplexer functions if it exists
if hc.multiplexer is not None:
    exp_starts += [[hc.multiplexer.all_off]]
    exp_ends += [[hc.multiplexer.all_off]]

hc.scheduler.add_exp(name=os.path.basename(FOLDER), starts=exp_starts, ends=exp_ends)

bar_gain = 0
tracker.start_time = time.time()
# first, let's add the experiments with a background
channel_vals = np.zeros((num_frames, 8), dtype=bool)
# alternate True between the 0-th and 1-st channel every 5 seconds = 5 * 120 frames
# channel_vals[..., 0] = np.arange(num_frames) % 360 < 180
# channel_vals[..., 1] = np.arange(num_frames) % 360 >= 180
# actually, instead of alternating, let's go 5 seconds of channel 0 and 20 seconds of channel 1
time_thresh = 5 * 120
channel_vals[:time_thresh, 0] = True
channel_vals[time_thresh:, 1] = True

# and lets' keep the motion gain to -1 until 15 seconds
motion_gains = np.zeros(num_frames)
# motion_gains[:15 * 120] = -1
# motion_gains[15 * 120:] = 0
motion_gains[:2 * 120] = -1
motion_gains[2 * 120:] = 0

print(backgrounds)

for (lbl, bg) in backgrounds.items():
    # randomly select a phase offset
    offset = np.random.random() * 2 * np.pi
    starts = [
        [bg.switch, True],
        [tracker.virtual_objects['bg'].set_motion_parameters, -1, offset],
        [hc.camera.import_config],
        [hc.camera.clear_headings],
        [hc.window.record_start],
        [set_attr_func, tracker, 'start_time', time.time],
        [print, f"bg: {lbl}"]
    ]
    middles = [
        [tracker.update_objects, hc.camera.update_heading],
        [hc.camera.get_background, hc.window.get_frame],
        [tracker.virtual_objects['bg'].update_motion_parameters, motion_gains],
        [bg.set_ry, tracker.virtual_objects['bg'].get_angle],
        # [bar.set_ry, tracker.virtual_objects['bar'].get_angle],
        [hc.window.record_frame]
    ]
    ends = [
        [bg.switch, False],
        [tracker.reset_virtual_object_motion],
        [tracker.add_test_data, hc.window.record_stop,
            {'start_test': getattr(tracker, 'start_time'), 'stop_test': time.time, 'bg': lbl}, True],
        [hc.window.reset_rot],
    ]
    if hc.multiplexer is not None:
        middles = [[hc.multiplexer.set_channels, channel_vals]] + middles
        ends = [[hc.multiplexer.set_channels, np.array([True, False, False, False, False, False, False, False])]] + ends
    hc.scheduler.add_test(num_frames, starts, middles, ends)
