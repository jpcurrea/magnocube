#! /usr/bin/env python
# run.py
import pyglet
import holocube.hc as hc
import os
import time

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
