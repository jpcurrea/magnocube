#! /usr/bin/env python
# run a series of exps with pyg and ard

import pyglet
from pyglet.window import key
from numpy import *
import numpy as np
import holocube.hc as hc
import os

num_frames = inf

# add horizontal and vertical gradients for testing the orientation of the viewports
xres = 256
# xvals = np.round(np.linspace(0, 255, xres)).astype('uint8')
xvals = np.arange(xres+1, dtype='uint8')
# horizontal gradient
arr = np.repeat(xvals[:, np.newaxis], xres, axis=-1)
arr = np.repeat(arr[:, :, np.newaxis], 4, axis=-1)
arr[..., -1] = 255
# from matplotlib import pyplot as plt
# plt.imshow(arr)
# plt.show()
# breakpoint()

# get the needed bottom and top values for the gradients
bottom, top = -np.arctan2(1, 2*np.sqrt(2)), np.arctan2(3, 2*np.sqrt(2))
h_grad_image = hc.stim.Quad_image(
    hc.window, left=0, right=2*np.pi, bottom=bottom, top=top, xres=xres, yres=xres, 
    xdivs=64, ydivs=1, dist=2)
h_grad_image.set_image(arr)

# shorthand key mods
sh = key.MOD_SHIFT
ct = key.MOD_CTRL
sc = key.MOD_CTRL + key.MOD_SHIFT

# add the experiment
hc.scheduler.add_exp()

starts =  [[hc.window.set_far,             3],
           [h_grad_image.switch,            True],
            # add key actions for rotating the observer
           [hc.window.add_keyhold_action,  key.LEFT, hc.window.inc_yaw,   .01],
           [hc.window.add_keyhold_action,  key.RIGHT, hc.window.inc_yaw, -.01],
           [hc.window.add_keyhold_action,  key.UP, hc.window.inc_pitch,   .01],
           [hc.window.add_keyhold_action,  key.DOWN, hc.window.inc_pitch, -.01],
           [hc.window.add_keyhold_action, (key.LEFT, ct), hc.window.inc_roll,   .01],
           [hc.window.add_keyhold_action, (key.RIGHT, ct), hc.window.inc_roll,   -.01]
          ]

middles = []

ends =    [[hc.window.set_far,                    2],
           [h_grad_image.switch,            False],
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


