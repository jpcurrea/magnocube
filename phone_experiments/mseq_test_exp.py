"""Test used to measure the temporal resolution of the display and therefore
it's limitations in transmitting information to both the observer and the 
DAQ. 

We're going to generate an m-sequence with 8 bits and a 2-bit shift register, then
we'll transmit the original sequence through a digital channel to the DAQ and then 
use this to set the indicator color and record this using a phototransducer to the DAQ.

We can then get the impulse and frequence response of the system by cross-correlating
the m-sequence with the recorded signal.
"""
from functools import partial
import holocube.hc as hc
from holocube.camera import Camera, TrackingTrial
import numpy as np
from math import ceil
import os
import time

FOLDER = 'mseq_test'
if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)
tracker = TrackingTrial(camera=hc.camera, window=hc.window, dirname=FOLDER)

exp_starts = [[hc.window.set_bg, [.5, .5, .5, 1.]],
              [tracker.h5_setup],
              [tracker.add_exp_attr, 'start_exp', time.time],
              [tracker.add_exp_attr, 'experiment', FOLDER],
              [hc.daq.start_recording, partial(tracker.__getattribute__, 'h5_file')],
              [hc.window.set_far, 5],
              [hc.window.set_ref, 0, (0, 0, 0)],
              [hc.camera.clear_headings]]
exp_ends = [[hc.window.set_bg, [0., 0., 0., 1.]],
            [hc.window.set_ref, 0, (0, 0, 0)],
            [hc.daq.stop_recording],
            [tracker.add_exp_attr, 'stop_exp', time.time],
            [tracker.save],
            [hc.camera.clear_headings]]
hc.scheduler.add_exp(name='mseq_test', starts=exp_starts, ends=exp_ends)

# get an msequence to set the ref colors
mseq = hc.tools.mseq(2, 8, 0, 0)
num_frames = len(mseq)
colors = np.zeros((num_frames, 3), dtype=int)
# when mseq == 1, set the ref color to white
colors[mseq == 1] = 255
# when mseq == -1, leave the ref color black

# elev_diff = np.append([0], np.diff(elev))
starts = [[hc.window.set_ref, 0, (0, 0, 0)],
          [hc.daq.write_to_channel, 'holostim_writer', 0, 'digital'],
          [hc.camera.clear_headings]]
middles = [[hc.camera.import_config],
        [hc.camera.get_background, hc.window.get_frame],
        [hc.daq.write_to_channel, 'holostim_writer', colors[:, 0], 'digital'],
        [hc.window.set_ref, 0, colors]]
ends = [[hc.window.set_ref, 0, (0, 0, 0)],
        [hc.daq.write_to_channel, 'holostim_writer', 0, 'digital'],
        [hc.window.reset_pos_rot],
        [hc.camera.reset_display_headings]]
hc.scheduler.add_test(num_frames, starts, middles, ends)