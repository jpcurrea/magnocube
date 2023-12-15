#! /usr/bin/env python
# run.py
import pyglet
import holocube.hc as hc
import holocube.camera as cameras
import os
import time

DUMMY = True
DUMMY_FN = 'H:\\Other computers\\My Computer\\pablo\\magnocube\\HQ_video\\2023_10_04_16_40_47.mp4'
# DUMMY_FN = 'H:\\Other computers\\My Computer\\pablo\\magnocube\\HQ_video\\2023_08_01_16_19_10.mp4'
WING_ANALYSIS = False

if DUMMY:
    hc.camera = cameras.Camera(window=hc.window, camera=DUMMY_FN,
                               wing_analysis=WING_ANALYSIS)

home_dir = os.getcwd()
hc.window.start(config_file='viewport.config')
hc.arduino.start('dummy')
hc.scheduler.start(hc.window, randomize=False, default_rest_time=.1, freq=120)
hc.scheduler.load_dir('experiments', suffix=('exp.py', 'rest.py'))
hc.scheduler.randomize = True
hc.camera.display_start()


hc.camera.kalman = True
hc.camera.capture_start()

# hc.scheduler.load_dir('H:/Other computers/My Computer/pablo/magnocube/experiments', suffix=('exp.py', 'rest.py'))
print('ready')

w = hc.window
s = hc.scheduler

# setup closing function for the whole program
@hc.window.event
def on_close():
    hc.camera.close()

pyglet.app.run()
