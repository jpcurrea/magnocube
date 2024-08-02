"""Test for the effect of bar motion at different azimuths.

Parameters:
-background: .5
-azimuths: 4 cardinal directions and their 4 midpoints = pi/4 * [0 - 7]
-contrast: -.5, +5
-time series: still for half, oscillating for the other
"""
from datetime import datetime
import holocube.hc as hc
import numpy as np
import os
from holocube.camera import TrackingTrial

def timestamp():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")

DATA_FOLDER = "bar_exp"
DURATION = 10
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

num_frames = DURATION * hc.scheduler.freq # 120 fps
# num_frames = np.inf

freq = .5   # oscillation frequency in Hz
ts = np.linspace(0, DURATION, round(num_frames/2))
orientations = np.zeros(num_frames)
orientations[round(num_frames/2.):] = 30 * np.sin(freq * ts * 2 * np.pi)
orientations *= np.pi / 180.

# keep track of data as we run experiments
tracker = TrackingTrial(camera=hc.camera, window=hc.window, dirname=DATA_FOLDER)
# experiment: add this experiment to the scheduler
# save_fn = os.path.join(FOLDER, str(timestamp()))
exp_starts = [[hc.window.set_far, 3],
              [hc.camera.storing_start, -1, DATA_FOLDER],
              [tracker.h5_setup],
              [tracker.store_camera_settings],
              [tracker.virtual_objects['fly_heading'].set_motion_parameters, -1],
              [hc.camera.clear_headings],
              [tracker.add_exp_attr, 'video_fn', hc.camera.get_save_fn],
              ]
exp_ends = [[hc.window.set_far,     1],
            [tracker.save],
            [hc.camera.storing_stop]]
hc.scheduler.add_exp(name=None, starts=exp_starts, ends=exp_ends)
# make orientation offsets
starting_oris = 45 * np.arange(8)
# make a dark and light bar
# set the visual width of the bar but I need to provide the actual width
dist = .8
angle = 30 * np.pi / 180.
width = 2 * dist * np.tan(angle/2)

stim_dark = hc.stim.Bars(hc.window, color=1., dist=dist, width=width)
stim_light = hc.stim.Bars(hc.window, color=1., dist=dist, width=width)
for ori in starting_oris:
    for stim, color in zip([stim_dark, stim_light], [0, 1]):
        for gain in [-1, 0]:
            oris = ori + orientations
            starts = [[stim.switch, True],
                      [hc.camera.import_config],
                      [tracker.virtual_objects['fly_heading'].set_motion_parameters, gain, hc.camera.update_heading],
                      [hc.camera.clear_headings],
                      [hc.window.record_start]]
            middles = [
                       [tracker.update_objects, 0],
                       [tracker.update_objects, hc.camera.update_heading],
                       #[hc.window.set_yaw, tracker.virtual_objects['fly_heading'].get_angle],
                       # [hc.window.set_yaw, 0],
                       # [stim.set_ry, np.pi/2],
                       [hc.window.record_frame]]
            ends = [[stim.switch, False],
                    [tracker.add_test_data, hc.window.record_stop,
                     {'gain': gain, 'color': color, 'start_angle': ori,
                      'orientation': oris * np.pi / 180.}, True],
                    [hc.window.reset_rot],
                    # [hc.camera.set_heading_gain, 0]
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
