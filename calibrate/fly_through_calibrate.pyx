# rest with a rotating bar

import holocube.hc as hc
from numpy import *
from pyglet.window import key

num_frames = inf

square = hc.stim.Movable_grating(hc.window, array([[-1,1,1,-1],[-1,-1,1,1], [-2,-2,-2,-2]]),
                                 sf=5, tf=3, o=pi/4, sd=.35)
pts = hc.stim.Points(hc.window, int(10**4), dims=[(-5, 5),(-5, 5),(-5, 5)], color=1, pt_size=4)

# add the experiment
hc.scheduler.add_exp()

starts =  [[hc.window.set_far,         3],
           [ hc.window.add_keyhold_action,  key.UP,                                 hc.window.inc_pitch,  .05],
           [ hc.window.add_keyhold_action,  key.DOWN,                               hc.window.inc_pitch, -.05],
           [ hc.window.add_keyhold_action,  key.LEFT,                               hc.window.inc_yaw,    .05],
           [ hc.window.add_keyhold_action,  key.RIGHT,                              hc.window.inc_yaw,   -.05],
           [ hc.window.add_keyhold_action, (key.LEFT,  key.MOD_SHIFT),              hc.window.inc_roll,    .05],
           [ hc.window.add_keyhold_action, (key.RIGHT, key.MOD_SHIFT),              hc.window.inc_roll,   -.05],
           [ hc.window.add_keyhold_action, (key.UP,    key.MOD_CTRL),               hc.window.inc_lift,    .05],
           [ hc.window.add_keyhold_action, (key.DOWN,  key.MOD_CTRL),               hc.window.inc_lift,   -.05],
           [ hc.window.add_keyhold_action, (key.LEFT,  key.MOD_CTRL),               hc.window.inc_slip,    .05],
           [ hc.window.add_keyhold_action, (key.RIGHT, key.MOD_CTRL),               hc.window.inc_slip,   -.05],
           [ hc.window.add_keyhold_action, (key.UP,    key.MOD_CTRL, key.MOD_SHIFT),hc.window.inc_thrust,-.05],
           [ hc.window.add_keyhold_action, (key.DOWN,  key.MOD_CTRL, key.MOD_SHIFT),hc.window.inc_thrust, .05],
           [ hc.window.add_keypress_action, key.END,                                hc.window.reset_pos_rot],
           [hc.arduino.set_lmr_scale,      -0.1],
           [square.set_ry,                  0],
           [square.switch,                  True],
           [pts.switch,                     True]]

middles = [[square.next_frame           ]]

ends =    [[square.switch,             False],
           [pts.switch,                False],
           [hc.window.remove_key_action,  key.UP],
           [hc.window.remove_key_action,  key.DOWN],
           [hc.window.remove_key_action,  key.LEFT],
           [hc.window.remove_key_action,  key.RIGHT],
           [hc.window.remove_key_action,  key.END],
           [hc.window.remove_key_action, (key.LEFT, key.MOD_SHIFT)],
           [hc.window.remove_key_action, (key.RIGHT, key.MOD_SHIFT)],
           [hc.window.remove_key_action, (key.UP, key.MOD_CTRL)],
           [hc.window.remove_key_action, (key.DOWN, key.MOD_CTRL)],
           [hc.window.remove_key_action, (key.LEFT, key.MOD_CTRL)],
           [hc.window.remove_key_action, (key.RIGHT, key.MOD_CTRL)],
           [hc.window.remove_key_action, (key.UP, key.MOD_CTRL, key.MOD_SHIFT)],
           [hc.window.remove_key_action, (key.DOWN, key.MOD_CTRL, key.MOD_SHIFT)],
           [hc.window.set_far,         2],
           [hc.window.reset_pos_rot]]

# add the test
hc.scheduler.add_test(num_frames, starts, middles, ends)

