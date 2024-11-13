"""
todo:
- add rotation and translational operations that are relative to the current
  orientation of the virtual camera
- fix the unwrapping function to reset whenever the 
    - I think I fixed it but I need to check for sure
"""

#! /usr/bin/env python
# run.py
import pyglet
import holocube.hc as hc
import holocube.camera as cameras
import os
import time

DUMMY = True
# DUMMY_FN = 'H:\\Other computers\\My Computer\\pablo\\magnocube\\HQ_video\\2023_10_04_16_40_47.mp4'
# DUMMY_FN = 'H:\\Other computers\\My Computer\\pablo\\magnocube\\HQ_video\\2023_08_01_16_19_10.mp4'
# DUMMY_FN = 'H:\\Other computers\\My Computer\\pablo\\magnocube\\flow_6_dof\\2024_03_13_11_39_05.mp4'
# DUMMY_FN = 'H:\\Other computers\\My Computer\\pablo\\magnocube\\thrust_OL_exp\\2024_03_13_12_40_50.mp4'
DUMMY_FN = 'test_60fps.mp4'
WING_ANALYSIS = False

if DUMMY:
    hc.camera = cameras.Camera(window=hc.window, camera=DUMMY_FN, wing_analysis=WING_ANALYSIS, com_correction=True)

home_dir = os.getcwd()
# hc.window.start(config_file='test_viewport.config')
hc.window.start(config_file='phone_viewport_test.config')
# hc.arduino.start('dummy')
hc.scheduler.start(hc.window, randomize=False, default_rest_time=.1, freq=30)
hc.scheduler.load_dir('test_experiments', suffix=('exp.py', 'rest.py'))
hc.scheduler.randomize = False
hc.camera.display_start()

#hc.window.inc_lift(1)

hc.camera.kalman = True
# hc.camera.capture_start()

# hc.scheduler.load_dir('H:/Other computers/My Computer/pablo/magnocube/experiments', suffix=('exp.py', 'rest.py'))
print('ready')

w = hc.window
s = hc.scheduler

# setup closing function for the whole program
@hc.window.event
def on_close():
    hc.camera.close()

pyglet.app.run()
