# pt_flow_exp.py
import holocube.hc as hc
from holocube.camera import TrackingTrial
import os
import numpy as np
from scipy import signal
import time
# import the partial function from the functools module
from functools import partial

# num_frames = np.inf
num_frames = int(round(5*hc.scheduler.freq))  # ~5 seconds
if num_frames % 2 == 1:
   num_frames += 1

# make a dot field
# pts = hc.stim.Points(hc.window, int(10**4), dims=[(-5, 5), (-5, 5), (-5, 5)], color=1, pt_size=4)
size = 1000
far = 5
dots_per_cube = 50 # dots per visible cube (far x far x far)
dot_density = dots_per_cube / ((2*far) ** 3)
# since the far cube fits into the total volume by the ratio (size/far)**3,
# the number of dots in the far cube is (size/far)**3 * dot_density
# let's decide on the total number of dots first, then calculate the size
num_dots = 5 * 10 ** 6
# instead, let's make pts thin along one dimension because we're only moving along each independently
# so volume = (2*size) * (2*size) * (2*far) = 8 * size ** 2 * far
# and num_dots = volume * dot_density => volume = num_dots / dot_density => 8 * size ** 2 * far = num_dots / dot_density
# so size = (num_dots / (far * dot_density)) ** (1/2)
volume = size * size * 2* far
size = (num_dots / (2 * far * dot_density)) ** (1/2)
pts = hc.stim.Points(hc.window, num_dots, dims=[(-size/2, size/2),(-far, far),(-size/2, size/2)], color=1, pt_size=5, method='random')
# bar = hc.stim.Bars(hc.window)

FOLDER = os.path.abspath("thrust_speed_exp")
if not os.path.exists(FOLDER):
   os.mkdir(FOLDER)

orig_bg = hc.window.bg_color
orig_near = .1
orig_far = 3
tracker = TrackingTrial(camera=hc.camera, window=hc.window, dirname=FOLDER)
# tracker.add_virtual_object(name='fly_heading', motion_gain=0, object=False)
# experiment: add this experiment to the scheduler
exp_starts = [[hc.window.set_far, far], 
              [hc.window.set_near, .1],
              [hc.window.set_bg, [0.0, 0.0, 0.0, 1.]],
              [hc.window.set_pos, [0, 0, 0]],
              [hc.camera.clear_headings],
              [hc.camera.storing_start, -1, FOLDER],
              [tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, hc.camera.update_heading],
              # add the experimental attributes
              [tracker.h5_setup],
              # video file
              [tracker.add_exp_attr, 'video_fn', hc.camera.get_save_fn],
              # experiment folder
              [tracker.add_exp_attr, 'exp_folder', FOLDER],
              # point coordinates
            #   [tracker.add_exp_attr, 'fly_heading', pts.coords],
              # experiment start time
              [tracker.add_exp_attr, 'start_exp', time.time],
              [pts.switch, True],
              ]

exp_ends = [[pts.switch, False],
            [tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, hc.camera.update_heading],
            [tracker.add_exp_attr, 'stop_exp', time.time],
            [tracker.save],
            [hc.camera.storing_stop],
            [hc.window.set_far, orig_far], 
            [hc.window.set_near, orig_near],
            [hc.window.set_bg, orig_bg],
            [hc.camera.clear_headings]
            ]
hc.scheduler.add_exp(name='thrust_speed_tuning', starts=exp_starts, ends=exp_ends)

velocity = -.3

for velocity in np.linspace(-.5, .5, 11, endpoint=True):
    for gain in [-1, 0]:
        # add the starts, middles, and ends for the test
        starts = [[hc.camera.reset_display_headings],
                [tracker.virtual_objects['fly_heading'].set_motion_parameters, gain, hc.camera.update_heading],
                [print, f"start angle= 0 pi\tvelocity={velocity:.2f}\tgain={gain}"],
                [tracker.add_exp_attr, 'start_test', time.time],
                [hc.window.record_start]
                ]
        middles = [[hc.camera.get_background, hc.window.get_frame],
                [tracker.update_objects, hc.camera.update_heading],
                # [tracker.update_objects, angles],
                [hc.window.set_rot_angles, tracker.virtual_objects['fly_heading'].get_angle],
                # [print, tracker.virtual_objects['fly_heading'].get_angle, hc.camera.get_heading],
                [hc.window.inc_thrust, velocity],
                # [hc.window.set_pos, tracker.virtual_objects['heading_subj'].get_position],
                [hc.window.record_frame]
                ]
        ends = [[pts.reset_pos_rot],
                [hc.camera.reset_display_headings],
                [tracker.reset_virtual_object_motion],
                [tracker.virtual_objects['fly_heading'].reset_position],
                [tracker.add_test_data, hc.window.record_stop, 
                {'optic_flow': 'thrust', 'start_angle': 0, 'velocity': velocity, 
                'start_test': partial(tracker.get_exp_attr, 'start_test'), 'stop_test': time.time,
                'fly_heading': tracker.virtual_objects['fly_heading'].get_angles, 
                'positions': tracker.virtual_objects['fly_heading'].get_positions}, True],
                [hc.window.reset_pos_rot]]
        hc.scheduler.add_test(num_frames, starts, middles, ends)