#! /usr/bin/env python
# run a series of exps with pyg and ard

import time
from pyglet.window import key
from numpy import *
import numpy as np
import holocube.hc as hc
import os
from holocube.camera import TrackingTrial

num_frames = inf
NAME = "optic_flow_velocity_tuning"
FOLDER = os.path.abspath(NAME)
CROSSHAIR = True
NUM_FRAMES = 60*20   # should be about 10 seconds
NUM_FRAMES /= 5
SPEED = .1
FAR = 5
far = FAR

# make a point field with appropriate density
size = 100
dots_per_cube = 200 # dots per visible cube (far x far x far)
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
pts = hc.stim.Points(hc.window, num_dots, dims=[(-size/2, size/2),(-far/2, far/2),(-size/2, size/2)], color=1, pt_size=5, method='random')
# make a TrackingTrial object to keep track of the virtual position of the points field
tracker = TrackingTrial(camera=hc.camera, window=hc.window, 
                        dirname=FOLDER)
tracker.add_virtual_object(name='pts', motion_gain=0,
                           start_angle=hc.camera.update_heading)

if CROSSHAIR:
    # make an image with a single 5 pixel crosshair
    xres = 256
    arr = np.zeros((xres, xres, 4), dtype='uint8')
    # thin vertical line
    arr[:, :2] = 255
    arr[:, -3:] = 255
    # short horizontal line
    arr[:1, 125:131, :] = 255
    arr[-1:, 125:131, :] = 255
    # get the needed bottom and top values for the image
    bottom, top = -np.arctan2(1, 2*np.sqrt(2)), np.arctan2(3, 2*np.sqrt(2))
    cross_hair_image = hc.stim.Quad_image(
        hc.window, left=0, right=2*np.pi, bottom=bottom, top=top, xres=xres, yres=xres, 
        xdivs=64, ydivs=1, dist=4)
    cross_hair_image.set_image(arr)
    tracker.add_virtual_object(name='crosshair', motion_gain=0,
                            start_angle=hc.camera.update_heading)

# add the experiment
exp_starts = [[hc.window.set_far, FAR],
              [hc.window.set_bg, [0.0, 0.0, 0.0, 0.0]],
              [tracker.h5_setup],
              [hc.camera.storing_start, -1, FOLDER, None, True],
              [tracker.store_camera_settings],
              [hc.camera.clear_headings],
              [pts.switch, True], 
              [tracker.add_exp_attr, 'video_fn', hc.camera.get_save_fn],
              [tracker.add_exp_attr, 'experiment', NAME],
              [tracker.add_exp_attr, 'start_exp', time.time],
            ]   
exp_ends = [[hc.window.set_far,     1],
            [hc.window.set_bg, [.5, .5, .5, 1.0]],
            [tracker.add_exp_attr, 'stop_exp', time.time],
            [tracker.save],
            [pts.switch, False],
            [hc.camera.storing_stop],
            [hc.camera.reset_display_headings],
            ]
              
if CROSSHAIR:
    exp_starts.append([cross_hair_image.switch, True])
    exp_ends.append([cross_hair_image.switch, False])

hc.scheduler.add_exp(name=NAME, starts=exp_starts, ends=exp_ends)

starts, middles, ends = [], [], []
# speeds = np.linspace(-2*SPEED, 2*SPEED, 9, endpoint=True)
speeds = np.array([-.4, -.2, -.1, -.05, 0, .05, .1, .2, .4])
# angles = np.zeros(NUM_FRAMES)
# angles[100:] = np.linspace(0, 12*np.pi, NUM_FRAMES - 100)
# angles = angles[None]
for rot_gain in [0, -1]:
    for trans_speed in speeds[speeds < 0]:
        starts =[
                # [pts.switch, True],
                [tracker.virtual_objects['pts'].set_motion_parameters, rot_gain, hc.camera.update_heading],
                # [tracker.virtual_objects['pts'].set_motion_parameters, rot_gain, 0.0],
                [tracker.virtual_objects['pts'].add_motion, None, trans_speed],
                # add the point field
                [print, f"rotation: {rot_gain}, translation speed: {trans_speed:.2f}, relative: True"]
                ]
        middles=[[hc.camera.get_background, hc.window.get_frame],
                [tracker.update_objects, hc.camera.update_heading],
                # [tracker.update_objects, np.pi/2],
                [pts.set_pos_rot, tracker.virtual_objects['pts'].get_pos_rot],
                ]

        ends = [
                # [pts.switch, False],
                [pts.reset_pos_rot],
                [tracker.reset_virtual_object_motion],
                [tracker.add_test_data, hc.window.record_stop,
                 {'rot_gain': rot_gain, 'thrust_speed': trans_speed, 'stop_test': time.time,
                  'pts_position': tracker.virtual_objects['pts'].get_positions, 
                  'relative_translation': True}, True],
                [hc.camera.clear_headings],
                ]
        if CROSSHAIR:
            starts.append([tracker.virtual_objects['crosshair'].set_motion_parameters, 0, hc.camera.update_heading])
            # starts.append([tracker.virtual_objects['crosshair'].set_motion_parameters, 0, 0])
            middles.append([cross_hair_image.set_pos_rot, tracker.virtual_objects['crosshair'].get_pos_rot])
            ends.append([cross_hair_image.reset_pos_rot])
        hc.scheduler.add_test(NUM_FRAMES, starts, middles, ends)

# num_frames = 5 * hc.scheduler.freq
num_frames = NUM_FRAMES
# pts = hc.stim.Points(hc.window, 1000, dims=[(-5, 5), (-5, 5), (-5, 5)], color=1, pt_size=4)
headings = np.linspace(0, 720 * np.pi / 180., 480)
headings = np.append(headings, headings[::-1])
speed = SPEED / 2
starts = [
        #   [pts.switch, True],
          [tracker.virtual_objects['pts'].set_motion_parameters, -1, hc.camera.update_heading],
          [tracker.virtual_objects['pts'].add_motion, headings, speed]
         ]
middles = [[hc.camera.import_config],
           [hc.camera.get_background, hc.window.get_frame],
           [tracker.update_objects, hc.camera.update_heading],
           [pts.set_pos_rot, tracker.virtual_objects['pts'].get_pos_rot],
          ]
ends = [[tracker.add_test_data, hc.window.record_stop,
            {'rot_gain': rot_gain, 'thrust_speed': speed, 'stop_test': time.time,
             'relative_translation': True, 'pts_position': tracker.virtual_objects['pts'].get_positions}, False],
        [tracker.reset_virtual_object_motion],
        [pts.reset_pos_rot],
        [hc.camera.clear_headings],]
if CROSSHAIR:
    starts.append([tracker.virtual_objects['crosshair'].set_motion_parameters, 0, hc.camera.update_heading])
    middles.append([cross_hair_image.set_pos_rot, tracker.virtual_objects['crosshair'].get_pos_rot])
    ends.append([cross_hair_image.reset_pos_rot])
# hc.scheduler.add_rest(num_frames, starts, middles, ends)