#! /usr/bin/env python
# run.py
import pyglet
import holocube.hc as hc
import os
import time

home_dir = os.getcwd()
hc.window.start(config_file='viewport.config')
hc.arduino.start('dummy')
hc.camera.display_start(buffer_fn="C:\\Users\\roach\\Desktop\\pablo\\arena\\_buffer.npy",
                        heading_fn="C:\\Users\\roach\\Desktop\\pablo\\arena\\_heading.npy")
hc.camera.capture_start()
time.sleep(.5)
# os.chdir(home_dir)

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
