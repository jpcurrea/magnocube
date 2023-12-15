#! /usr/bin/env python
# run.py
import pyglet
import holocube.hc as hc
import os
import time

# hc.camera = hc.cameras.Camera(window=hc.window, camera="./HQ_video/2023_08_01_16_19_10.mp4")
hc.camera = hc.cameras.Camera(window=hc.window, camera="./HQ_video/2023_10_04_16_40_47.mp4")

home_dir = os.getcwd()
hc.window.start(config_file='test_viewport.config')
hc.arduino.start('dummy')

hc.camera.kalman = True
hc.camera.display_start()

hc.scheduler.start(hc.window, randomize=False, default_rest_time=.1, freq=60)
hc.scheduler.load_dir('experiments', suffix=('exp.py', 'rest.py'))
    # hc.scheduler.load_dir('experiments', suffix=('exp.py', 'rest.py'))
hc.scheduler.randomize = True
print('ready')

w = hc.window
s = hc.scheduler

# setup closing function for the whole program
@hc.window.event
def on_close():
    hc.camera.close()

pyglet.app.run()
