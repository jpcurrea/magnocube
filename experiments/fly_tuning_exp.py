# grating_sf_tuning_exp.py
import holocube.hc as hc
import numpy as n
import numpy as np
from holocube.camera import TrackingTrial

# 1 second per test
num_frames = np.inf
# num_frames = 1200

# make a dot field
pts = hc.stim.Points(hc.window, 1000, dims=[(-5, 5), (-5, 5), (-5, 5)], color=1, pt_size=4)

orig_bg = hc.window.bg_color
# experiment: add this experiment to the scheduler
DATA_FOLDER = "./"
tracker = TrackingTrial(camera=hc.camera, window=hc.window, dirname=DATA_FOLDER)
# tracker.add_virtual_object(name='bg', motion_gain=0, object=True)
exp_starts = [[hc.window.set_far, 5],
              [hc.window.set_bg, [0., 0., 0., 1.]],
              [tracker.virtual_objects['fly_heading'].set_motion_parameters, -1, 0],
              [hc.camera.clear_headings],
              ]
exp_ends = [[hc.window.set_far, 1],
            [hc.window.set_bg, orig_bg],
            [hc.camera.clear_headings]]
hc.scheduler.add_exp(name='fly tuning', starts=exp_starts, ends=exp_ends)


# generate a series of heading positions between -90 and 90
# angles = n.linspace(-90, 90, )
headings = np.linspace(0, 720 * np.pi / 180., 480)
headings = np.append(headings, headings[::-1])

# add a test different heading locations with a loop
starts = [[pts.switch, True],
          [hc.camera.reset_display_headings],
          [tracker.virtual_objects['fly_heading'].add_motion, headings]
         ]
middles = [[hc.camera.import_config],
           [hc.camera.get_background, hc.window.get_frame],
           [tracker.update_objects, hc.camera.update_heading],
           # [hc.window.set_rot, np.linspace(0, 2 * np.pi, 100)[:, None]],
        #    [print, tracker.virtual_objects['fly_heading'].get_angle],
           [hc.window.set_rot, tracker.virtual_objects['fly_heading'].get_rot],
        #    [hc.window.set_yaw, tracker.virtual_objects['fly_heading'].get_angle],
          ]
ends = [[pts.switch, False],
        [hc.camera.reset_display_headings],
        [hc.window.reset_rot]]
hc.scheduler.add_test(num_frames, starts, middles, ends)

# add the rest
# rest_frames = 120
# bar = hc.stim.Bars(hc.window)
#
# starts = [[bar.switch, True]]
# middles = [[hc.window.inc_yaw, hc.arduino.lmr]]
# ends = [[bar.switch, False],
#         [hc.window.reset_rot]]
# hc.scheduler.add_rest(rest_frames, starts, middles, ends)
 
