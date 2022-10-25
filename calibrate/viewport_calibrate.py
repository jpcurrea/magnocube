#! /usr/bin/env python
# run a series of exps with pyg and ard

import pyglet
from pyglet.window import key
from numpy import *
import holocube.hc as hc
import os

num_frames = inf

s = hc.stim.sphere_lines_class(hc.window)

# print(os.path.split(os.path.abspath(__file__)))
front_image = hc.stim.Quad_image(hc.window, left=-pi/4, right=pi/4, bottom=-pi/4, top=pi/4, xdivs=24, ydivs=1, dist=1.5)
front_image.load_image('./calibrate/front.png')
left_image = hc.stim.Quad_image(hc.window, left=-pi/4, right=pi/4, bottom=-pi/4, top=pi/4, xdivs=24, ydivs=1, dist=1.5)
left_image.load_image('./calibrate/left.png')
left_image.inc_ry(90)
right_image = hc.stim.Quad_image(hc.window, left=-pi/4, right=pi/4, bottom=-pi/4, top=pi/4, xdivs=24, ydivs=1, dist=1.5)
right_image.load_image('./calibrate/right.png')
right_image.inc_ry(-90)
back_image = hc.stim.Quad_image(hc.window, left=-pi/4, right=pi/4, bottom=-pi/4, top=pi/4, xdivs=24, ydivs=1, dist=1.5)
back_image.load_image('./calibrate/back.png')
back_image.inc_ry(180)
bottom_image = hc.stim.Quad_image(hc.window, left=-pi/4, right=pi/4, bottom=-pi/4, top=pi/4, xdivs=24, ydivs=1, dist=1.4)
bottom_image.load_image('./calibrate/bottom.png')
bottom_image.inc_rx(-90)
top_image = hc.stim.Quad_image(hc.window, left=-pi/4, right=pi/4, bottom=-pi/4, top=pi/4, xdivs=24, ydivs=1, dist=1.5)
top_image.load_image('./calibrate/top.png')
top_image.inc_rx(90)

# shorthand key mods
sh = key.MOD_SHIFT
ct = key.MOD_CTRL
sc = key.MOD_CTRL + key.MOD_SHIFT

# add the experiment
hc.scheduler.add_exp()

starts =  [[hc.window.set_far,             3],
           [front_image.switch,            True],
           [right_image.switch,            True],
           [left_image.switch,             True],
           [bottom_image.switch,           True],
           [top_image.switch,              True],
           [back_image.switch,             True],
           [s.switch,                      True],
           [hc.window.viewport_inc_ind,    0],
           [hc.window.add_keypress_action, key.TAB,         hc.window.viewport_inc_ind],
           [hc.window.add_keyhold_action,  key.UP,          hc.window.viewport_set_val, 'bottom',   1, 'increment'],
           [hc.window.add_keyhold_action,  key.DOWN,        hc.window.viewport_set_val, 'bottom',  -1, 'increment'],
           [hc.window.add_keyhold_action,  key.LEFT,        hc.window.viewport_set_val, 'left',    -1, 'increment'],
           [hc.window.add_keyhold_action,  key.RIGHT,       hc.window.viewport_set_val, 'left',     1, 'increment'],
           [hc.window.add_keyhold_action, (key.UP, sh),     hc.window.viewport_set_val, 'height',   1, 'increment'],
           [hc.window.add_keyhold_action, (key.DOWN, sh),   hc.window.viewport_set_val, 'height',  -1, 'increment'],
           [hc.window.add_keyhold_action, (key.LEFT, sh),   hc.window.viewport_set_val, 'width',   -1, 'increment'],
           [hc.window.add_keyhold_action, (key.RIGHT, sh),  hc.window.viewport_set_val, 'width',    1, 'increment'],
           [hc.window.add_keyhold_action, (key.UP, ct),     hc.window.viewport_set_val, 'tilt',     1, 'increment'],
           [hc.window.add_keyhold_action, (key.DOWN, ct),   hc.window.viewport_set_val, 'tilt',    -1, 'increment'],
           [hc.window.add_keyhold_action, (key.LEFT, ct),   hc.window.viewport_set_val, 'pan',     -1, 'increment'],
           [hc.window.add_keyhold_action, (key.RIGHT, ct),  hc.window.viewport_set_val, 'pan',      1, 'increment'],
           [hc.window.add_keyhold_action, (key.LEFT, sc),   hc.window.viewport_set_val, 'dutch',   -1, 'increment'],
           [hc.window.add_keyhold_action, (key.RIGHT, sc),  hc.window.viewport_set_val, 'dutch',    1, 'increment'],
           [hc.window.add_keypress_action, key.X,           hc.window.viewport_set_val, 'scalex',   1, 'increment'],
           [hc.window.add_keypress_action, key.Y,           hc.window.viewport_set_val, 'scaley',   1, 'increment'],
           [hc.window.add_keypress_action, key.ENTER,       hc.window.save_config,      'test_viewport.config']
]

middles = []

ends =    [[hc.window.set_far,                    2],
           [front_image.switch,                   False],
           [right_image.switch,                   False],
           [left_image.switch,                    False],
           [bottom_image.switch,                  False],
           [top_image.switch,                     False],
           [back_image.switch,                    False],
           [s.switch,                             False],
           [hc.window.remove_key_action,        key.TAB],
           [hc.window.remove_key_action,        key.UP],
           [hc.window.remove_key_action,        key.DOWN],
           [hc.window.remove_key_action,        key.LEFT],
           [hc.window.remove_key_action,        key.RIGHT],
           [hc.window.remove_key_action,       (key.UP, sh)],
           [hc.window.remove_key_action,       (key.DOWN, sh)],
           [hc.window.remove_key_action,       (key.LEFT, sh)],
           [hc.window.remove_key_action,       (key.RIGHT, sh)],
           [hc.window.remove_key_action,       (key.UP, ct)],
           [hc.window.remove_key_action,       (key.DOWN, ct)],
           [hc.window.remove_key_action,       (key.LEFT, ct)],
           [hc.window.remove_key_action,       (key.RIGHT, ct)],
           [hc.window.remove_key_action,       (key.LEFT, sc)],
           [hc.window.remove_key_action,       (key.RIGHT, sc)],
           [hc.window.remove_key_action,        key.X],
           [hc.window.remove_key_action,        key.Y],
           [hc.window.remove_key_action,        key.ENTER],
           [hc.window.viewport_set_val,        'bg', [0.0, 0.0, 0.0, 1.0], 'set', 'all']
]


hc.scheduler.add_test(num_frames, starts, middles, ends)


