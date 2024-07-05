# pt_flow_exp.py
import holocube.hc as hc
import numpy as n

# 2 seconds per test
num_frames = n.inf

# make a dot field
# pts = hc.stim.Points(hc.window, int(10**4), dims=[(-5, 5), (-5, 5), (-5, 5)], color=1, pt_size=4)
pts = hc.stim.Points(hc.window, int(10**4), dims=[(-5, 5),(-5, 5),(-5, 5)], color=1, pt_size=4)
bar = hc.stim.Bars(hc.window)

# experiment: add this experiment to the scheduler
exp_starts = [[hc.window.set_near, .01],
              [hc.window.set_far, 3]]
exp_ends = [[hc.window.set_near, .1],
            [hc.window.set_far, 1]]
exp_starts = [[hc.window.set_far, 3]]
exp_ends = [[hc.window.set_far, 1]]
hc.scheduler.add_exp(name=None, starts=exp_starts, ends=exp_ends)
# hc.scheduler.add_exp()

# test1: add a test to the experiment
starts = [[hc.window.set_bg, [0., 0., 0., 0.]],
          [pts.switch, True],
          [bar.switch, True],
          [ hc.window.add_keyhold_action,  'up',                  hc.window.inc_pitch,  .05],
          [ hc.window.add_keyhold_action,  'down',                hc.window.inc_pitch, -.05],
          [ hc.window.add_keyhold_action,  'left',                hc.window.inc_yaw,    .05],
          [ hc.window.add_keyhold_action,  'right',               hc.window.inc_yaw,   -.05],
          [ hc.window.add_keyhold_action,  'shift left',          hc.window.inc_roll,    .05],
          [ hc.window.add_keyhold_action,  'shift right',         hc.window.inc_roll,   -.05],
          [ hc.window.add_keyhold_action,  'shift up',            hc.window.inc_lift,    .05],
          [ hc.window.add_keyhold_action,  'shift down',          hc.window.inc_lift,   -.05],
          [ hc.window.add_keyhold_action,  'ctrl left',           hc.window.inc_slip,    .05],
          [ hc.window.add_keyhold_action,  'ctrl right',          hc.window.inc_slip,   -.05],
          [ hc.window.add_keyhold_action,  'shift ctrl up',       hc.window.inc_thrust,-.05],
          [ hc.window.add_keyhold_action,  'shift ctrl down',     hc.window.inc_thrust, .05],
          [ hc.window.add_keypress_action, 'end',                 hc.window.reset_pos_rot],
          [ hc.window.set_yaw, -n.pi/4]
          ] 
# middles = [[hc.window.inc_thrust, -.02]]
middles = []
ends = [[pts.switch, False],
        [bar.switch, False],
        [hc.window.remove_key_actions, ('up', 'down', 'left', 'right', 'end', 'f',
                                        'shift up', 'shift down', 'shift left',
                                        'shift right', 'ctrl left', 'ctrl right',
                                        'shift ctrl up', 'shift ctrl down')],
        [hc.window.reset_pos]]
hc.scheduler.add_test(num_frames, starts, middles, ends)

# add the rest
rest_frames = 120

starts = [[bar.switch, True]]
middles = [[hc.window.inc_yaw, hc.arduino.lmr]]
ends = [[bar.switch, False],
        [hc.window.reset_rot]]
hc.scheduler.add_rest(rest_frames, starts, middles, ends)
