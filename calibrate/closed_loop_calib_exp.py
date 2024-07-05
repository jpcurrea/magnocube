#! /usr/bin/env python
# run a series of exps with pyg and ard

import pyglet
from pyglet.window import key
from numpy import *
import numpy as np
import holocube.hc as hc
import os
from holocube.camera import TrackingTrial

num_frames = inf

# make an image with a single 5 pixel crosshair
xres = 256
arr = np.zeros((xres, xres, 4), dtype='uint8')
# thin vertical line
arr[:, :2] = 255
arr[:, -3:] = 255
# short horizontal line
# arr[:1, 125:131, :] = 255
# arr[-1:, 125:131, :] = 255

# get the needed bottom and top values for the image
bottom, top = -np.arctan2(1, 2*np.sqrt(2)), np.arctan2(3, 2*np.sqrt(2))
cross_hair_image = hc.stim.Quad_image(
    hc.window, left=0, right=2*np.pi, bottom=bottom, top=top, xres=xres, yres=xres, 
    xdivs=64, ydivs=1, dist=2, )
cross_hair_image.set_image(arr)

# make a point field with appropriate density
size = 100
far = 5
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
# pts = hc.stim.Points(hc.window, num_dots, dims=[(-size/2, size/2),(-far/2, far/2),(-size/2, size/2)], color=1, pt_size=5, method='random')
pts = hc.stim.Points(hc.window, num_dots, dims=[(-size/2, size/2),(-far/2, 0),(-size/2, size/2)], color=1, pt_size=5, method='random')

# make a TrackingTrial object to keep track of the virtual position of the points field
tracker = TrackingTrial(camera=hc.camera, window=hc.window, 
                        dirname='closed_loop_calib_exp')
tracker.add_virtual_object(name='pts', motion_gain=0,
                           start_angle=hc.camera.update_heading)
tracker.add_virtual_object(name='crosshair', motion_gain=0,
                           start_angle=hc.camera.update_heading)

# shorthand key mods
sh = key.MOD_SHIFT
ct = key.MOD_CTRL
sc = key.MOD_CTRL + key.MOD_SHIFT

# add the experiment
hc.scheduler.add_exp()

delta_angle = 0.05
delta_pos = 0.05
starts =  [[hc.window.set_far,             far],
           [hc.window.set_bg,              [0.0, 0.0, 0.0, 1]],
           [cross_hair_image.switch,            True],
            # add key actions for rotating the observer
           [hc.window.add_keyhold_action,  key.LEFT, hc.window.inc_yaw,   delta_angle],
           [hc.window.add_keyhold_action,  key.RIGHT, hc.window.inc_yaw,  -delta_angle],
            # let's limit rotation to the horizontal plane
           [hc.window.add_keyhold_action,  'ctrl up', hc.window.inc_pitch,   delta_angle],
           [hc.window.add_keyhold_action,  'ctrl down', hc.window.inc_pitch, -delta_angle],
        #    [hc.window.add_keyhold_action, (key.LEFT, ct), hc.window.inc_roll,   delta_angle],
        #    [hc.window.add_keyhold_action, (key.RIGHT, ct), hc.window.inc_roll,  -delta_angle],
            # add key actions for translating the observer
        #    [hc.window.add_keyhold_action,  'shift up', hc.window.inc_thrust,    -delta_pos],
        #    [hc.window.add_keyhold_action,  'shift down', hc.window.inc_thrust,  delta_pos],
           [hc.window.add_keyhold_action,  'shift left', hc.window.inc_slip,  delta_pos],
           [hc.window.add_keyhold_action,  'shift right', hc.window.inc_slip, -delta_pos],
            # add the point field
           [pts.switch, True], 
           [tracker.virtual_objects['pts'].set_motion_parameters, 0, hc.camera.update_heading],
           [tracker.virtual_objects['pts'].add_motion, None, delta_pos],
           [tracker.virtual_objects['crosshair'].set_motion_parameters, 0, hc.camera.update_heading],
          ]

middles = [[hc.camera.get_background, hc.window.get_frame],
        #    [tracker.update_objects, hc.camera.update_heading],
        #    [pts.set_pos_rot, tracker.virtual_objects['pts'].get_pos_rot],
        #    [cross_hair_image.set_pos_rot, tracker.virtual_objects['crosshair'].get_pos_rot],
]

ends =    [[hc.window.set_far,                    2],
           [hc.window.set_bg,              [0.0, 0.0, 0.0, 1]],
           [cross_hair_image.switch,            False],
           [pts.switch, False],           
           [hc.window.remove_key_action,        key.TAB],
           [hc.window.remove_key_action,        key.UP],
           [hc.window.remove_key_action,        key.DOWN],
           [hc.window.remove_key_action,        key.LEFT],
           [hc.window.remove_key_action,        key.RIGHT],
           [hc.window.remove_key_action,       (key.LEFT, ct)],
           [hc.window.remove_key_action,       (key.RIGHT, ct)],
           [hc.window.reset_pos_rot],
]


hc.scheduler.add_test(num_frames, starts, middles, ends)


