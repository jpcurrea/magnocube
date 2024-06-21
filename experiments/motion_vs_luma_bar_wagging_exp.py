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
from matplotlib import pyplot as plt

def set_attr_func(object, var, func):
    val = func()
    setattr(object, var, val)

def timestamp():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")

FOLDER = os.path.abspath("motion vs luma bar wagging")
DURATION = 10
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)

# make background black
num_frames = DURATION * hc.scheduler.freq

# keep track of data as we run experiments
hc.camera.update_heading()
tracker = TrackingTrial(camera=hc.camera, window=hc.window, dirname=FOLDER)
tracker.add_virtual_object(name='bar', motion_gain=0,
                           start_angle=hc.camera.update_heading)
tracker.add_virtual_object(name='bg', motion_gain=-1,
                           start_angle=hc.camera.update_heading)

# Make the bar and background stimuli
sequence_length = 2**9
# xres = 96
xres = sequence_length
pad = .25
bottom, top = - np.arctan2(1, 2*np.sqrt(2)), np.arctan2(3, 2*np.sqrt(2))
cyl = hc.stim.Quad_image(hc.window, left= 0, right=2 * pi, bottom=bottom,
                         top=top, xres=xres,
                         yres=xres, xdivs=64, ydivs=1, dist=2)
cyl_gray = hc.stim.Quad_image(hc.window, left= 0, right=2 * pi, bottom=bottom,
                              top=top, xres=xres,
                              yres=xres, xdivs=64, ydivs=1, dist=2)
bar = hc.stim.Quad_image(hc.window, left=0*pi, right=2*pi, 
                         bottom=bottom, top=top, xres=xres, yres=xres, xdivs=64, ydivs=1,
                         dist=1)


# make a random period gratingx
# width, height = 512, 512
width, height = xres, xres
order = math.ceil(np.log2(width))
which_seq = np.random.randint(100)
mseq = hc.tools.mseq(2, order, whichSeq=which_seq)
mseq = mseq[:width]

arr = np.zeros((height, width, 4), dtype='uint8')
arr[..., -1] = 255
# arr[:, 1:][:, mseq == 1, 2] = 255
arr[:, 1:][:, mseq == 1] = 255
# arr[:] = 255
arr[..., :2] = 0
# arr[:] = 0
cyl.set_image(np.copy(arr))
arr[..., :3] = 128
arr[..., :2] = 0
arr[..., -1] = 255
cyl_gray.set_image(arr)
# grab a random 30 degrees of arr as the bar texture
# width corresponds to 360 degrees
# bar_width/heading = width/360 => bar_width = heading * width / 360
bar_angle = 30 * np.pi / 180.                  # rads
bar_width = int(round(bar_angle * width / (2 * np.pi)))
# make the bar width even so that it's symmetrical about the origin
if bar_width % 2 == 1:
    bar_width += 1
# the possible starting points for the bar are 0 - (width - bar_width)
bar_starts = np.arange(width - bar_width)
bar_start = np.random.choice(bar_starts)
# make another msequence for the bar stimulus
which_seq = np.random.randint(100)
mseq2 = hc.tools.mseq(2, order, whichSeq=which_seq)
mseq2 = mseq2[:width]
bar_vals = mseq[bar_start:bar_start + bar_width]
bar_arr = np.zeros((height, width, 4), dtype='uint8')
# add values for the lower bound
lower_bound = int(-bar_width/2)
upper_bound = int(bar_width/2)
# rotate by 3/4 revolution
dist = int(round(.25 * xres))
lower_bound += dist
upper_bound += dist
bar_arr[:, lower_bound:upper_bound, 2][:, bar_vals == 1] = 255
bar_arr[:, lower_bound:upper_bound, 3] = 255                    # alpha
bar.set_image(bar_arr)

# prep for using an msequence for the orientations
ts = np.linspace(0, DURATION, num_frames)
orientations = np.zeros(num_frames)
start_frame = round(num_frames/3.)
# find power of 2 closest to the duration of motion
# power = ceil(np.log2(len(ts[start_frame:])))
# start_frame = int(len(ts) - 2**power)

# OR use a triangle wave to control the relative position of the bar
num_frames = DURATION * hc.scheduler.freq
# num_frames = 1200
freq = 2   # oscillation frequency in Hz
amplitude = 180 * np.pi / 180. # oscillation amplitude
amplitude /= 2.    # note: the sawtooth oscillates between +/- amplitude
ts = np.linspace(0, DURATION, num_frames)
orientations = np.zeros(num_frames)
start_frame = np.argmax(np.where(ts <= 1)[0])
sawtooth = signal.sawtooth(freq * ts[:-start_frame] * 2*np.pi, .5)
sawtooth -= -1
orientations[start_frame:] = amplitude * sawtooth
orientations += np.pi/2

orientations_left = orientations - np.pi
orientations_right = - orientations

# define test parameters
exp_starts = [[hc.window.set_far, 3],
              [hc.window.set_bg, [0., 0., 0., 0.]],
              [tracker.h5_setup],
              [hc.camera.storing_start, -1, FOLDER, None, True],
              [tracker.store_camera_settings],
            #   [tracker.virtual_objects['bar'].set_motion_parameters, 0, hc.camera.update_heading],
              [tracker.virtual_objects['bar'].set_motion_parameters, 0, 0],
              [hc.camera.clear_headings],
              [tracker.add_exp_attr, 'video_fn', hc.camera.get_save_fn],
              [tracker.add_exp_attr, 'experiment', os.path.basename(FOLDER)],
              [tracker.add_exp_attr, 'motion_bar_vals', bar_vals],
              [tracker.add_exp_attr, 'bg_vals', mseq],
              [tracker.add_exp_attr, 'orientations', orientations],
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
tracker.start_time = 0
# first, let's add the experiments with a background
for bg, bg_lbl in zip([cyl, cyl_gray], ['random', 'gray']):
    for bg_gain in [-1, 0]:
        for bar_orientations, eye_side in zip([orientations_right, orientations_left], ['right', 'left']):
            starts = [
                [bar.switch, True],
                [bg.switch, True],
                [hc.camera.import_config],
                [tracker.virtual_objects['bg'].set_motion_parameters, bg_gain, hc.camera.update_heading],
                [tracker.virtual_objects['bar'].add_motion, bar_orientations],
                [hc.camera.clear_headings],
                [hc.window.record_start],
                [print, f"bg_texture={bg_lbl}\tbar_side={eye_side}"],
                [set_attr_func, tracker, 'start_time', time.time]
            ]
            middles = [
                [hc.camera.get_background, hc.window.get_frame],
                [tracker.update_objects, hc.camera.update_heading],
                # [print, tracker.virtual_objects['bar'].get_angle, hc.camera.get_heading],
                [bar.set_ry, tracker.virtual_objects['bar'].get_angle],
                # [bar.set_ry, hc.camera.get_heading],
                [hc.window.record_frame]
            ]
            ends = [
                [bg.switch, False],
                [bar.switch, False],
                [tracker.reset_virtual_object_motion],
                [tracker.add_test_data, hc.window.record_stop,
                 {'bg_texture': bg_lbl, 'side': eye_side, 'bg_gain': bg_gain,
                  'start_test': getattr(tracker, 'start_time'), 'stop_test': time.time,
                  'bar_position': tracker.virtual_objects['bar'].get_angles}, True],
                [hc.window.reset_rot],
            ]
            hc.scheduler.add_test(num_frames, starts, middles, ends)