"""Test for the effect of bar motion at different azimuths.

Parameters:
-background: .5
-azimuths: 4 cardinal directions and their 4 midpoints = pi/4 * [0 - 7]
-contrast: -.5, +5
-time series: still for half, oscillating for the other
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

def timestamp():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")

DATA_FOLDER = "fourier feedback"
DURATION = 6
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

# make background black
# num_frames = DURATION * 60 # 120 fps
# num_frames = DURATION * 60 # 60 fps
num_frames = DURATION * hc.scheduler.freq
# num_frames = np.inf

freq = 2   # oscillation frequency in Hz
amplitude = 30 * np.pi / 180. # oscillation amplitude
amplitude /= 2.
ts = np.linspace(0, DURATION, num_frames)
orientations = np.zeros(num_frames)
start_frame = round(num_frames/3.)

# orientations[start_frame:] = 30 * np.sin(freq * ts[:-start_frame] * 2 * np.pi)
sawtooth = signal.sawtooth(freq * ts[:-start_frame] * 2*np.pi + np.pi/2., .5)
orientations[start_frame:] = amplitude * sawtooth
# orientations = np.linspace(0, 2*np.pi, width)

# keep track of data as we run experiments
hc.camera.update_heading()
tracker = TrackingTrial(camera=hc.camera, window=hc.window, dirname=DATA_FOLDER)
tracker.add_virtual_object(name='bar', motion_gain=-1,
                           start_angle=hc.camera.update_heading, object=True)
tracker.add_virtual_object(name='bg', motion_gain=-1,
                           start_angle=hc.camera.update_heading, object=True)
# experiment: add this experiment to the scheduler
# save_fn = os.path.join(FOLDER, str(timestamp()))
# cyl = hc.stim.Quad_image(hc.window, left=-1*pi, right=1*pi, bottom=-.2*pi, top=.2*pi, xres=512,
#                          yres=512, xdivs=64, ydivs=1, dist=2)
# bar = hc.stim.Quad_image(hc.window, left=-1*pi, right=1*pi, bottom=-.2*pi, top=.2*pi, xres=512,
#                          yres=512, xdivs=64, ydivs=1)
sequence_length = 2**7
# xres = 96
xres = sequence_length
cyl = hc.stim.Quad_image(hc.window, left= 0, right=2 * pi, bottom=-.2*pi,
                         top=.2*pi, xres=xres,
                         yres=xres, xdivs=64, ydivs=1, dist=2)
bar = hc.stim.Quad_image(hc.window, left=0*pi, right=2*pi, bottom=-.2*pi,
                         top=.2*pi, xres=xres,
                         yres=xres, xdivs=64, ydivs=1)


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
# arr[..., :2] = 0
# arr[:] = 0
cyl.set_image(arr)
# grab a random 30 degrees of arr as the bar texture
# width corresponds to 360 degrees
# bar_width/heading = width/360 => bar_width = heading * width / 360
bar_angle = 30 * np.pi / 180.                  # rads
bar_width = round(bar_angle * width / (2 * np.pi))
# make the bar width even so that it's symmetrical about the origin
if bar_width % 2 == 1:
    bar_width += 1
# the possible starting points for the bar are 0 - (width - bar_width)
bar_starts = np.arange(width - bar_width)
bar_start = np.random.choice(bar_starts)
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
bar_arr[:, lower_bound:upper_bound, 3] = 255                                       # alpha
# bar_arr[:, :bar_width, 2][:, bar_vals == 1] = 255
# bar_arr[:, :bar_width, 3] = 255                                       # alpha
# bar_arr[:, lower_bound:, 2][:, bar_vals[:lower_bound] == 1] = 255  # blue
# bar_arr[:, lower_bound:, 3] = 255                                       # alpha
# # and for the upper bound
# bar_arr[:, :upper_bound, 2][:, bar_vals[lower_bound:] == 1] = 255
# bar_arr[:, :upper_bound, 3] = 255
# bar_stop = bar_start + bar_width
# bar_arr[:, :bar_width, 2][:, bar_vals == 1] = 255
# bar_arr[:, 1:][:, :bar_width, :3]= 255
# bar_arr[:, 1:][:, :bar_width, 3] = 255
# todo: find out necessary offset to place the bar at midline
# bar_arr[:, bar_start : bar_stop, :] = 255
# make bar only blue
# bar_arr[..., :2] = 0
bar.set_image(bar_arr)
# rotate bar by np.pi/2 - bar_angle/2


# setup experimental variables
# stds = [0, 4]
# make orientation offsets
# starting_oris = (np.pi / 4) * np.arange(8)
starting_oris = (np.pi / 6) * np.arange(12)

exp_starts = [[hc.window.set_far, 3],
              [hc.window.set_bg, [0., 0., 0., 1.]],
              [hc.camera.storing_start, -1, DATA_FOLDER, None, True],
              [tracker.h5_setup],
              [tracker.store_camera_settings],
              [tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, hc.camera.update_heading],
              [tracker.virtual_objects['bar'].set_motion_parameters, -1, hc.camera.update_heading],
              # [tracker.virtual_objects['bar'].set_motion_parameters, 0, hc.camera.update_heading],
              [hc.camera.clear_headings],
              [tracker.add_exp_attr, 'video_fn', hc.camera.get_save_fn],
              [tracker.add_exp_attr, 'experiment', DATA_FOLDER],
              [tracker.add_exp_attr, 'bar_vals', bar_vals],
              [tracker.add_exp_attr, 'bg_vals', mseq],
              [tracker.add_exp_attr, 'start_exp', time.time],
              ]
exp_ends = [[hc.window.set_far,     1],
            [hc.window.set_bg, [.5, .5, .5, 1.0]],
            [tracker.add_exp_attr, 'stop_exp', time.time],
            [tracker.save],
            [hc.camera.storing_stop],
            ]
hc.scheduler.add_exp(name=DATA_FOLDER, starts=exp_starts, ends=exp_ends)



offset = np.pi
for ori in starting_oris:
    for bar_gain in [-1, 0]:
        for bg_gain in [-1, 0]:
            oris = ori + orientations
            # oris = orientations
            starts = [[bar.switch, True],
                      [cyl.switch, True],
                      [hc.camera.import_config],
                      [tracker.virtual_objects['bar'].set_motion_parameters, bar_gain, hc.camera.update_heading],
                      [tracker.virtual_objects['bg'].set_motion_parameters, bg_gain, hc.camera.update_heading],
                      # [tracker.virtual_objects['bar'].set_motion_parameters, bar_gain, 0],
                      # [tracker.virtual_objects['bg'].set_motion_parameters, bg_gain, 0],
                      [tracker.virtual_objects['bar'].add_motion, oris],
                      #[tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, hc.camera.update_heading],
                      [hc.camera.clear_headings],
                      [hc.window.record_start],
                      # [print, f"bar at {ori * 180 / np.pi :.3f} degs"]
                      ]
            middles = [
                       [tracker.update_objects, hc.camera.update_heading],
                       # [tracker.update_headings, 0],
                       [bar.set_ry, tracker.virtual_objects['bar'].get_angle],
                       [cyl.set_ry, tracker.virtual_objects['bg'].get_angle],
                       # [bar.inc_ry, np.pi/180.],
                       # [bar.set_ry, oris],
                       # [cyl.set_ry, tracker.virtual_objects['fly_heading'].get_heading],
                       [hc.window.record_frame]
                       ]
            ends = [[cyl.switch, False],
                    [bar.switch, False],
                    [tracker.virtual_objects['bar'].clear_motion],
                    [tracker.add_test_data, hc.window.record_stop,
                     {'bar_gain': bar_gain, 'bg_gain': bg_gain,
                      'start_angle': 0, 'orientation': oris,
                      'bar': tracker.virtual_objects['bar'].get_angles,
                      'stop_test': time.time}, True],
                    [hc.window.reset_rot],
                    ]
            hc.scheduler.add_test(num_frames, starts, middles, ends)

for bg_gain in [-1, 0]:
    starts = [[bar.switch, False],
              [cyl.switch, True],
              [hc.camera.import_config],
              [tracker.virtual_objects['bar'].set_motion_parameters, 0, hc.camera.update_heading],
              [tracker.virtual_objects['bg'].set_motion_parameters, bg_gain, hc.camera.update_heading],
              # [tracker.virtual_objects['bar'].set_motion_parameters, gain, 0],
              # [tracker.virtual_objects['bg'].set_motion_parameters, 0, 0],
              [tracker.virtual_objects['bg'].add_motion, orientations],
              #[tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, hc.camera.update_heading],
              [hc.camera.clear_headings],
              [hc.window.record_start],
              [print, f"background with {bg_gain} gain"]
              ]
    middles = [
               [tracker.update_objects, hc.camera.update_heading],
               # [tracker.update_headings, 0],
               [bar.set_ry, tracker.virtual_objects['bar'].get_angle],
               [cyl.set_ry, tracker.virtual_objects['bg'].get_angle],
               # [bar.inc_ry, np.pi/180.],
               # [bar.set_ry, oris],
               # [cyl.set_ry, tracker.virtual_objects['fly_heading'].get_heading],
               [hc.window.record_frame]
               ]
    ends = [[cyl.switch, False],
            [bar.switch, False],
            [tracker.virtual_objects['bg'].clear_motion],
            [tracker.add_test_data, hc.window.record_stop,
             {'bar_gain': np.nan, 'bg_gain': bg_gain,
              'start_angle': np.nan, 'orientation': orientations,
              'bar': tracker.virtual_objects['bar'].get_angles,
              'stop_test': time.time}, True],
            [hc.window.reset_rot],
            ]
    hc.scheduler.add_test(num_frames, starts, middles, ends)

# add the rest
# rest_frames = 120
# bar = hc.stim.Bars(hc.window)
#
# starts = [[bar.switch, True]]
# middles = [[hc.window.inc_yaw, hc.arduino.lmr]]
# ends = [[bar.switch, False],
#         [hc.window.reset_rot]]
# hc.scheduler.add_rest(rest_frames, starts, middles, ends)
