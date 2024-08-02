# sf test, try a series of spatial frequencies to get a curve, left right
import holocube.hc as hc
from numpy import *

# count exps
num_frames = 330
start_frame = 30

# a grating stimulus
#f_sp_l = hc.stim.Grating_cylinder(hc.window, 0, fast=True, half=True)
gratings = []

# a series of contrasts
# a series of sfs
isf = 25*pi/180 # inverted sf
sf = 1./isf
contrasts = linspace(0, 1, 10)
tf = 10.
orientations = array([0, pi])           # the orientation of the 2d grating

side = sqrt(2)
top_position = array([[-1,1,1,-1],[0,0,side,side], [-side,-side,0,0]])  #the x, y, and z positions of the 4 corners, with the fly at (0, 0, 0)
middle_position = array([[-1,1,1,-1],[-1,-1,1,1], [-1,-1,-1,-1]])
bottom_position = array([[-1,1,1,-1],[0,0,-side,-side], [-side,-side,0,0]])
positions = array([middle_position])    #the 3d position of the 2d grating

for c in contrasts:
    for ori in orientations:
        for position in positions:
            tfa = zeros((num_frames))
            tfa[start_frame:] = tf
            gratings += [hc.stim.Movable_grating(
                hc.window, 
                position,
                o=ori,
                c=c,
                sf=sf, tf=tfa, fast=True, sd=.35)]
#    f_sp_l.add_grating(sf=sf, tf=tfa, c=1.0, maxframes=3*num_frames)#, sdb=.35)

#anim_seq = arange(num_frames)

contrast_seqs = []
for con_ind in range(len(contrasts)):
    seq = hc.tools.test_num_flash(con_ind+1, num_frames)
#    seq = append(array([(0, 0, 0)]), seq)
    contrast_seqs.append(seq)

ori_seqs = []
for ori_ind in range(len(orientations)):
    seq = hc.tools.test_num_flash(ori_ind+2, num_frames)
#    seq = append(array([(0, 0, 0)]), seq)
#    seq[1] = (0, 255, 0)
    ori_seqs.append(seq)

pos_seqs = []
for pos_ind in range(len(positions)):
    seq = hc.tools.test_num_flash(pos_ind+3, num_frames)
#    seq = append(array([(0, 0, 0)]), seq)
#    seq[1] = (0, 255, 0)
    pos_seqs.append(seq)

# import pdb; pdb.set_trace()

# add the experiment
hc.scheduler.add_exp()
for con_num, con_seq in enumerate(contrast_seqs):
    for ori_num, ori_seq in enumerate(ori_seqs):
        for pos_num, pos_seq in enumerate(pos_seqs):
            ind = con_num*len(orientations)*len(positions) + ori_num*len(positions) + pos_num
            grating = gratings[ind]
            starts =  [[hc.window.set_far,                 3],
                       [hc.window.set_bg,                 [0.0,0.0,0.0,1.0]],
                       [grating.set_ry,                    0   ],
                       [grating.switch,                    True],
                       [grating.on,                        True]]

            middles = [[hc.window.set_ref,                 0, con_seq],
                       [hc.window.set_ref,                 1, ori_seq],
                       [hc.window.set_ref,                 2, pos_seq],
                       [grating.next_frame                     ]]

            ends =    [[grating.switch,                    False],
                       [hc.window.set_far,                 2]]

                    # add each test
            hc.scheduler.add_test(num_frames, starts, middles, ends)

# add the rest
num_frames = 200
rbar = hc.stim.cbarr_class(hc.window, dist=1)

starts =  [[hc.window.set_far,         2],
           [hc.window.set_bg,          [0.0,0.0,0.0,1.0]],
           [hc.arduino.set_lmr_scale,  -.1],
           [rbar.set_ry,               0],
           [rbar.switch,               True] ]
middles = [[rbar.inc_ry,               hc.arduino.lmr]]
ends =    [[rbar.switch,               False],
           [hc.window.set_far,         2]]
 
hc.scheduler.add_rest(num_frames, starts, middles, ends)
