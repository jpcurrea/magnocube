"""Test for the effect of revolving a motion defined bar on a moving background.

Parameters:
-background: high resolution random pattern grating 
-bar_starts: 4 cardinal directions
-bg motion: revolve at a constant velocity from [-360, -180, -90, 0, 90, 180, 360. for the full experiment (3 seconds)
-bar motion: still for 1 second, then revolve at a constant velocity (-360, -180, -90, 0, 90, 180, 360)
= 7 * 7 * 4 * 3 s = 9.8 minutes
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

FOLDER = "revolving_fbar"
FOLDER = os.path.abspath(FOLDER)
DURATION = 10
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
sequence_length = 2**9
# xres = 96
xres = sequence_length
cyl = hc.stim.Quad_image(hc.window, left= 0, right=2 * pi, bottom=-.2*pi,
                         top=.2*pi, xres=xres,
                         yres=xres, xdivs=64, ydivs=1, dist=2)
motion_bar = hc.stim.Quad_image(hc.window, left=0*pi, right=2*pi, bottom=-.2*pi,
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
# arr[:, 1:][:, mseq == 1, 2] = 255
arr[:, 1:][:, mseq == 1] = 255
# arr[:] = 255
# arr[..., :2] = 0
# arr[:] = 0
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
bar_arr[:, lower_bound:upper_bound, :3][:, bar_vals == 1] = 255
bar_arr[:, lower_bound:upper_bound, 3] = 255                                  # alpha
# plt.imsave('test_img.png', bar_arr)
motion_bar.set_image(bar_arr)
cyl.set_image(arr)

# prep for using an msequence for the orientations
ts = np.linspace(0, DURATION, num_frames)
orientations = np.zeros(num_frames)
start_frame = round(num_frames/3.)
# find power of 2 closest to the duration of motion
# power = ceil(np.log2(len(ts[start_frame:])))
# start_frame = int(len(ts) - 2**power)

# OR use a triangle wave to control the relative position of the bar
num_frames = DURATION * hc.scheduler.freq
freq = 2   # oscillation frequency in Hz
# amplitude = 30 * np.pi / 180. # oscillation amplitude
amplitude = np.pi # oscillation amplitude
amplitude /= 2.
ts = np.linspace(0, DURATION, num_frames)
orientations = np.zeros(num_frames)
start_frame = int(round(num_frames/3.))
sawtooth = signal.sawtooth(freq * ts[:-start_frame] * 2*np.pi + np.pi/2., .5)
orientations[start_frame:] = amplitude * sawtooth

# define test parameters
bg_gains = np.array([-1, 0])                     # closed and open loop
bg_velocities = np.array([-135 * np.pi / 180., 0, 135 * np.pi / 180.])
# starting_oris = np.pi/6 * np.arange(12)          # 12 evenly distributed starting locations
# starting_oris = np.pi/6 * np.arange(12)          # 12 evenly distributed starting locations
# starting_oris = np.pi/2 * np.array([0, 1, 2, 3])
starting_oris = np.array([np.pi])
bar_velocities = np.array([-np.pi, np.pi])       # rads/s
# => 2 bar vel. x 3 bg vel. x 2 bg gains x 12 starting orientations
# = 144 tests * 3 s = 7.2 minutes


exp_starts = [[hc.window.set_far, 5],
              [hc.window.set_bg, [0., 0., 0., 0.]],
              [hc.camera.storing_start, -1, FOLDER, None, True],
              [tracker.h5_setup],
              [tracker.store_camera_settings],
              [tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, hc.camera.update_heading],
              [tracker.virtual_objects['bar'].set_motion_parameters, 0, hc.camera.update_heading],
              [hc.camera.clear_headings],
              [tracker.add_exp_attr, 'video_fn', hc.camera.get_save_fn],
              [tracker.add_exp_attr, 'experiment', FOLDER],
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
hc.scheduler.add_exp(name=os.path.basename(FOLDER).replace("_", " "), starts=exp_starts, ends=exp_ends)

bar_gain = 0
tracker.start_time = 0
for bg_gain in [-1, 0]:
    for bar_gain in [-1, 0]:
        for bar_velocity in bar_velocities:
            # calculate the necessary orientations based on the bar and bg velocities
            bar_oris = np.arange(num_frames) * bar_velocity / hc.scheduler.freq
            bar_oris += np.pi
            starts = [[motion_bar.switch, True],
                    [cyl.switch, True],
                    [hc.camera.import_config],
                    [tracker.virtual_objects['bg'].set_motion_parameters, bg_gain, hc.camera.update_heading],
                    [tracker.virtual_objects['bar'].set_motion_parameters, bar_gain, hc.camera.update_heading],
                    [tracker.virtual_objects['bar'].add_motion, bar_oris],
                    [hc.camera.clear_headings],
                    [hc.window.record_start],
                    [set_attr_func, tracker, 'start_time', time.time]
                    ]
            middles = [
                    [hc.camera.get_background, hc.window.get_frame],
                    [tracker.update_objects, hc.camera.update_heading],
                    [motion_bar.set_ry, tracker.virtual_objects['bar'].get_angle],
                    [cyl.set_ry, tracker.virtual_objects['bg'].get_angle],
                    [hc.window.record_frame]
                    ]
            ends = [[cyl.switch, False],
                    [motion_bar.switch, False],
                    [tracker.virtual_objects['bar'].clear_motion],
                    [tracker.virtual_objects['bg'].clear_motion],
                    [tracker.add_test_data, hc.window.record_stop,
                    {'bar_gain': bar_gain, 'bg_gain': bg_gain, 'bar_velocity': bar_velocity,
                    'bar_orientation': tracker.virtual_objects['bar'].get_angles,
                    'bg_orientation': tracker.virtual_objects['bg'].get_angles,
                    'start_test': getattr(tracker, 'start_time'),
                    'stop_test': time.time}, True],
                    [hc.window.reset_rot],
                    ]
            hc.scheduler.add_test(num_frames, starts, middles, ends)

# for bg_gain in [-1, 0]:
#     starts = [[bar.switch, False],
#               [cyl.switch, True],
#               [hc.camera.set_ring_params],
#               [tracker.virtual_objects['bar'].set_motion_parameters, 0, hc.camera.update_heading],
#               [tracker.virtual_objects['bg'].set_motion_parameters, bg_gain, hc.camera.update_heading],
#               # [tracker.virtual_objects['bar'].set_motion_parameters, gain, 0],
#               # [tracker.virtual_objects['bg'].set_motion_parameters, 0, 0],
#               [tracker.virtual_objects['bg'].add_motion, orientations],
#               #[tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, hc.camera.update_heading],
#               [hc.camera.clear_headings],
#               [hc.window.record_start],
#               [print, f"background with {bg_gain} gain"]
#               ]
#     middles = [
#                [tracker.update_objects, hc.camera.update_heading],
#                # [tracker.update_headings, 0],
#                [bar.set_ry, tracker.virtual_objects['bar'].get_angle],
#                [cyl.set_ry, tracker.virtual_objects['bg'].get_angle],
#                # [bar.inc_ry, np.pi/180.],
#                # [bar.set_ry, oris],
#                # [cyl.set_ry, tracker.virtual_objects['fly_heading'].get_heading],
#                [hc.window.record_frame]
#                ]
#     ends = [[cyl.switch, False],
#             [bar.switch, False],
#             [tracker.virtual_objects['bg'].clear_motion],
#             [tracker.add_test_data, hc.window.record_stop,
#              {'bar_gain': np.nan, 'bg_gain': bg_gain,
#               'start_angle': np.nan, 'orientation': orientations,
#               'bar': tracker.virtual_objects['bar'].get_angles,
#               'stop_test': time.time}, True],
#             [hc.window.reset_rot],
#             ]
#     hc.scheduler.add_test(num_frames, starts, middles, ends)

# add the rest
# rest_frames = 120
# bar = hc.stim.Bars(hc.window)
#
# starts = [[bar.switch, True]]
# middles = [[hc.window.inc_yaw, hc.arduino.lmr]]
# ends = [[bar.switch, False],
#         [hc.window.reset_rot]]
# hc.scheduler.add_rest(rest_frames, starts, middles, ends)
