"""Parameters:
-background: high resolution random pattern grating which moves when no bar is
 present
-bar azimuths: 4 cardinal directions
-time series: still for 2 seconds, msequence for 4
-with or without blur: low pass filter of <5 degrees
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

FOLDER = "WF feedback"
DURATION = 6
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)

# make background black
# num_frames = DURATION * 60 # 120 fps
# num_frames = DURATION * 60 # 60 fps
num_frames = DURATION * hc.scheduler.freq
# num_frames = np.inf



# keep track of data as we run experiments
hc.camera.update_heading()
tracker = TrackingTrial(camera=hc.camera, window=hc.window, dirname=FOLDER)
tracker.add_virtual_object(name='bg', motion_gain=-1,
                           start_angle=hc.camera.update_heading, object=True)
sequence_length = 2**7
xres = sequence_length
cyl = hc.stim.Quad_image(hc.window, left= 0, right=2 * pi, bottom=-.2*pi,
                         top=.2*pi, xres=xres,
                         yres=xres, xdivs=64, ydivs=1, dist=2)
# make a random period gratingx
# width, height = 512, 512
width, height = xres, xres
order = math.ceil(np.log2(width))
which_seq = np.random.randint(100)
mseq = hc.tools.mseq(2, order, whichSeq=which_seq)
mseq = mseq[:width]

arr = np.zeros((height, width, 4), dtype='uint8')
arr[..., -1] = 255
arr[:, 1:][:, mseq == 1, 2] = 255
cyl.set_image(arr)
# prep for using an msequence for the orientations
# find power of 2 closest to the duration of motion
ts = np.linspace(0, DURATION, num_frames)
start_frame = round(num_frames/3.)
power = int(ceil(np.log2(num_frames - start_frame)))
start_frame = int(len(ts) - 2**power)
orientations = np.zeros(num_frames)

# OR use a triangle wave to control the relative position of the bar
num_frames = DURATION * hc.scheduler.freq
freq = 2   # oscillation frequency in Hz
amplitude = 30 * np.pi / 180. # oscillation amplitude
amplitude /= 2.
ts = np.linspace(0, DURATION, num_frames)
orientations = np.zeros(num_frames)
# start_frame = round(num_frames/3.)
triangle = amplitude * signal.sawtooth(freq * ts[:-start_frame] * 2*np.pi + np.pi/2., .5)
sawtooth_left = amplitude * signal.sawtooth(freq * 2 * ts[:-start_frame] * 2*np.pi + np.pi/2., 0)
sawtooth_right = amplitude * signal.sawtooth(freq * 2 * ts[:-start_frame] * 2*np.pi + np.pi/2., 1)
# make orientation offsets
exp_starts = [[hc.window.set_far, 3],
              [hc.window.set_bg, [0., 0., 0., 1.]],
              [hc.camera.storing_start, -1, FOLDER, None, True],
              [tracker.h5_setup],
              [tracker.store_camera_settings],
              [tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, hc.camera.update_heading],
              [hc.camera.clear_headings],
              [tracker.add_exp_attr, 'video_fn', hc.camera.get_save_fn],
              [tracker.add_exp_attr, 'experiment', FOLDER],
              [tracker.add_exp_attr, 'bg_vals', mseq],
              [tracker.add_exp_attr, 'start_exp', time.time],
              ]
exp_ends = [[hc.window.set_far,     1],
            [hc.window.set_bg, [.5, .5, .5, 1.0]],
            [tracker.add_exp_attr, 'stop_exp', time.time],
            [tracker.save],
            [hc.camera.storing_stop],
            ]
hc.scheduler.add_exp(name=FOLDER, starts=exp_starts, ends=exp_ends)


bg_gains = [-4, -2, -1, -.5, 0, .5, 1, 2, 4]
tracker.start_time = time.time()
for bg_gain in bg_gains:
    mseq = np.cumsum(hc.tools.mseq(2, power))
    max_dev = abs(mseq).max()
    mseq *= amplitude/max_dev
    mseq = np.append([0], mseq)
    for orientations in [triangle, sawtooth_left, sawtooth_right, mseq]:
        oris = np.zeros(len(ts))
        oris[start_frame:] = orientations
        starts = [[cyl.switch, True],
                  [hc.camera.set_ring_params],
                  [tracker.virtual_objects['bg'].set_motion_parameters, bg_gain, hc.camera.update_heading],
                  [tracker.virtual_objects['bg'].add_motion, oris],
                  [hc.camera.clear_headings],
                  [hc.window.record_start],
                  [tracker.record_timestamp]
                  ]
        middles = [
                   [tracker.update_objects, hc.camera.update_heading],
                   [cyl.set_ry, tracker.virtual_objects['bg'].get_angle],
                   [hc.window.record_frame]
                   ]
        ends = [[cyl.switch, False],
                [tracker.virtual_objects['bg'].clear_motion],
                [tracker.add_test_data, hc.window.record_stop,
                 {'bg_gain': bg_gain, 'bg_motion' : tracker.virtual_objects['bg'].get_angles, 
                  'orientation': oris, 'start_test': tracker.get_timestamp,
                  'stop_test': time.time}, True],
                [hc.window.reset_rot],
                ]
        hc.scheduler.add_test(num_frames, starts, middles, ends)