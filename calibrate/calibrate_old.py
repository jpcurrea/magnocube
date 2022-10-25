#! /usr/bin/env python
# run a series of exps with pyg and ard

import pyglet
from pyglet.window import key
import holocube.hc5 as hc

# ardname = '/dev/ttyACM0'
ardname = 'dummy'

project = 0
bg = [0.3, 0.3, 0.3, 1.]
# bg = [1., 1., 1., 1.]
near, far = .01, 100.

randomize = 0


s = hc.stim.sphere_lines_class(hc.window, add=True)
l = hc.stim.lines_class(hc.window, 19, add=False, color=0, ln_width=12)
# F
l.coords[:,0] = [-.5,-.5,-.7]
l.coords[:,1] = [-.5, .5,-.7]
l.coords[:,2] = [-.5, .5,-.7]
l.coords[:,3] = [ .5, .5,-.7]
l.coords[:,4] = [-.5, .1,-.7]
l.coords[:,5] = [ .3, .1,-.7]
# L
l.coords[:,6] = [-.7, .5,-.5]
l.coords[:,7] = [-.7,-.5,-.5]
l.coords[:,8] = [-.7,-.5,-.5]
l.coords[:,9] = [-.7,-.5, .5]
#R
l.coords[:,10] = [ .7, .5, .5]
l.coords[:,11] = [ .7,-.5, .5]
l.coords[:,12] = [ .7, .5, .5]
l.coords[:,13] = [ .7, .5,-.3]
l.coords[:,14] = [ .7, .1, .5]
l.coords[:,15] = [ .7, .1,-.3]
l.coords[:,16] = [ .7, .5,-.3]
l.coords[:,17] = [ .7, .1,-.3]
l.coords[:,18] = [ .7, .1, .0]
l.coords[:,19] = [ .7,-.5,-.3]
#T
l.coords[:,20] = [ .0, .7, .5]
l.coords[:,21] = [ .0, .7,-.5]
l.coords[:,22] = [-.5, .7,-.5]
l.coords[:,23] = [ .5, .7,-.5]
#B
l.coords[:,24] = [-.5, -.7, .5]
l.coords[:,25] = [-.5, -.7,-.5]
l.coords[:,26] = [-.5, -.7,-.5]
l.coords[:,27] = [ .5, -.7,-.5]
l.coords[:,28] = [-.5, -.7, .5]
l.coords[:,29] = [ .5, -.7, .5]
l.coords[:,30] = [-.5, -.7, .0]
l.coords[:,31] = [ .5, -.7,-.5]
l.coords[:,32] = [-.5, -.7, .0]
l.coords[:,33] = [ .5, -.7, .5]
#X rear
l.coords[:,34] = [-.5, -.5, .7]
l.coords[:,35] = [ .5,  .5, .7]
l.coords[:,36] = [ .5, -.5, .7]
l.coords[:,37] = [-.5,  .5, .7]

l.add()

# start the components
hc.window.start(project=project, bg_color=bg, near=near, far=far, config='viewport_config.txt')

# hc.arduino.start(ardname)

# add some keys
hc.window.add_keypress_action(key.A,  hc.window.vps.add, hc.window.world)
hc.window.add_keypress_action((key.A, key.MOD_SHIFT),  hc.window.vps.remove)
hc.window.add_keypress_action(key._1, hc.window.vps.choose_vp, 1)
hc.window.add_keypress_action(key._2, hc.window.vps.choose_vp, 2)
hc.window.add_keypress_action(key._3, hc.window.vps.choose_vp, 3)
hc.window.add_keypress_action(key._4, hc.window.vps.choose_vp, 4)
hc.window.add_keypress_action(key._5, hc.window.vps.choose_vp, 5)
hc.window.add_keypress_action(key._6, hc.window.vps.choose_vp, 6)
hc.window.add_keypress_action(key._7, hc.window.vps.choose_vp, 7)
hc.window.add_keypress_action(key._8, hc.window.vps.choose_vp, 8)
hc.window.add_keypress_action(key._9, hc.window.vps.choose_vp, 9)
hc.window.add_keypress_action((key._1, key.MOD_CTRL), hc.window.vps.choose_ref_pt, 1)
hc.window.add_keypress_action((key._2, key.MOD_CTRL), hc.window.vps.choose_ref_pt, 2)
hc.window.add_keypress_action((key._3, key.MOD_CTRL), hc.window.vps.choose_ref_pt, 3)
hc.window.add_keypress_action((key._4, key.MOD_CTRL), hc.window.vps.choose_ref_pt, 4)

hc.window.add_keypress_action(key.UP, hc.window.vps.move, bottom=1)
hc.window.add_keypress_action(key.DOWN, hc.window.vps.move, bottom=-1)
hc.window.add_keypress_action(key.LEFT, hc.window.vps.move, left=-1)
hc.window.add_keypress_action(key.RIGHT, hc.window.vps.move, left=1)
hc.window.add_keypress_action((key.UP, key.MOD_SHIFT), hc.window.vps.move, bottom=10)
hc.window.add_keypress_action((key.DOWN, key.MOD_SHIFT), hc.window.vps.move, bottom=-10)
hc.window.add_keypress_action((key.LEFT, key.MOD_SHIFT), hc.window.vps.move, left=-10)
hc.window.add_keypress_action((key.RIGHT, key.MOD_SHIFT), hc.window.vps.move, left=10)
hc.window.add_keypress_action((key.UP, key.MOD_ALT), hc.window.vps.move, height=1)
hc.window.add_keypress_action((key.DOWN, key.MOD_ALT), hc.window.vps.move, height=-1)
hc.window.add_keypress_action((key.LEFT, key.MOD_ALT), hc.window.vps.move, width=-1)
hc.window.add_keypress_action((key.RIGHT, key.MOD_ALT), hc.window.vps.move, width=1)
hc.window.add_keypress_action((key.UP, key.MOD_ALT + key.MOD_SHIFT), hc.window.vps.move, height=10)
hc.window.add_keypress_action((key.DOWN, key.MOD_ALT + key.MOD_SHIFT), hc.window.vps.move, height=-10)
hc.window.add_keypress_action((key.LEFT, key.MOD_ALT + key.MOD_SHIFT), hc.window.vps.move, width=-10)
hc.window.add_keypress_action((key.RIGHT, key.MOD_ALT + key.MOD_SHIFT), hc.window.vps.move, width=10)

hc.window.add_keypress_action(key.X, hc.window.vps.change_axis)
hc.window.add_keypress_action((key.X, key.MOD_SHIFT), hc.window.vps.change_up)
hc.window.add_keypress_action((key.X, key.MOD_CTRL), hc.window.vps.change_scale, scale=1)

hc.window.add_keypress_action(key.P, hc.window.print_keypress_actions)

hc.window.add_keypress_action(key.ENTER, hc.window.vps.save_file)

hc.window.add_keypress_action(key.R, hc.window.vps.make_ref_window)

# no need to start the scheduler

w = hc.window

w.print_keypress_actions()

# run pyglet
pyglet.app.run()

