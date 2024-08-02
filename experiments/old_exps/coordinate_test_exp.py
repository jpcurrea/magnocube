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

FOLDER = "H:\\Other computers\\My Computer\\pablo\\magnocube\\test"
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)
# NUM_FRAMES = DURATION * hc.scheduler.freq
# START_FRAME = 1 * hc.scheduler.freq
NUM_FRAMES = 5 * 60

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
# arr[..., -1] = 255
# arr[:, 1:][:, mseq == 1, 2] = 255
# replace arr with half white, half black
# arr[:, :round(width/2)] = 255
# arr[:, round(width/2):] = 0
arr[:, 0] = 255

cyl.set_image(arr)
bgs = [cyl]

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


# oris = np.linspace(0, 20*np.pi, NUM_FRAMES)
oris = np.zeros(NUM_FRAMES)
for gain in [-2, -1, 0, 1, 2]:
    starts = [[cyl.switch, True],
            [hc.camera.import_config],
            [tracker.virtual_objects['fly_heading'].set_motion_parameters, gain, hc.camera.update_heading],
            [tracker.virtual_objects['bg'].add_motion, oris],
            [hc.camera.clear_headings],
            [hc.window.record_start],
            [tracker.record_timestamp]]
    middles = [
                [tracker.update_objects, hc.camera.update_heading],
                [cyl.set_ry, tracker.virtual_objects['bg'].get_angle],
                [hc.window.record_frame]
                ]
    ends = [[cyl.switch, False],
            [tracker.virtual_objects['bg'].clear_motion],
            [tracker.add_test_data, hc.window.record_stop,
                {'bg_orientation' : tracker.virtual_objects['bg'].get_angles, 
                'orientation': oris, 'start_test': tracker.get_timestamp,
                'stop_test': time.time}, True],
            [hc.window.reset_rot],
            ]
    hc.scheduler.add_test(NUM_FRAMES, starts, middles, ends)