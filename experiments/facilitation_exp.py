"""Pat Shoemaker's stimulus for testing facilitation.

Designate an area of  60° x 60° in the receptive field of one eye, and define four parallel tracks with 20° separation between them. Under the assumption that LC11 is used in the detection of walking flies, maybe these should be horizontal tracks and the direction of motion should be away from the midline. Define a set of segment lengths for the different components of the experiment, from the full 60° length down to some value smaller than the LC11 receptive field – say, 60°, 30°, 20°, and 10°?  We’ll have to program those sequences appropriately – in the dragonfly experiments we specified that each jump between segments must move at least two tracks away laterally, and I think the starting points of the segments were randomized in the direction of motion (I wrote the Python script to generate the stimulus, but I’m barely literate in Python and I don’t remember exactly what I did 12 years ago!) I’m not sure that both constraints can be met with only 4 tracks, but we can work that out. During the experiment, each component is run with uninterrupted motion (i.e., never any delays between segments), but we imposed a significant rest period between different components to prevent habituation.

"""
import holocube.hc as hc
import numpy as np
from math import ceil
from holocube.camera import Camera


dot = hc.stim.Sphere()

# Define the tracks
max_tracks = 4
track_width = 60   # degrees; represents the horizontal range of motion
track_height = 60    # degrees; the vertical range of tracks
# we want the dot to rotate at ~200 degrees per second
# and it should traverse 60 degrees, so we need to know how many frames
speed = 200  # degrees per second
duration = track_width / speed
num_frames = int(duration * 60)  # 60 fps
# make sure that num_frames is divisible by max_tracks
num_frames += max_tracks - num_frames % max_tracks
# find the next value above num_frames that is divisible by 2, 3, and 4
num_frames = 12 * ceil(num_frames / 12.)

xvals = np.linspace(-track_width / 2, track_width / 2, num_frames)
ytracks = np.linspace(-track_height / 2, track_height / 2, max_tracks, endpoint=True)

order = [0, 2, 3, 1]

yvals = []
for num_tracks in range(max_tracks):
    num_tracks += 1
    ys = ytracks[order[:num_tracks]]
    # now repeat and add the yvals so that there are num_frames of yvals
    yvals += [np.repeat(ys, num_frames // num_tracks)]

## plot the 4 different tests
# from matplotlib import pyplot as plt
# fig, axes = plt.subplots(1, max_tracks, figsize=(12, 3))
# for ax, ys in zip(axes, yvals):
#     # add thin gray lines for all of the tracks
#     for y in ytracks:
#         ax.axhline(y, color='gray', lw=0.5)
#     # and plot the yvals, splitting them into different lines with a square at the start and arrow at the end of 
#     # each line
#     y_set = np.unique(ys)
#     for y in y_set:
#         inds = ys == y
#         ax.plot(xvals[inds], ys[inds], 'k')
#         ax.plot(xvals[inds][0], y, 'ks')
#         ax.plot(xvals[inds][-1], y, 'k>')
# plt.tight_layout()
# plt.show() 

exp_starts = [[hc.window.set_bg, [.5, .5, .5, 1.]],
              [hc.camera.clear_headings]] 
exp_ends = [[hc.window.set_bg, [0., 0., 0., 1.]],
            [hc.camera.clear_headings]]
hc.scheduler.add_exp(name='facilitation', starts=exp_starts, ends=exp_ends)

yvals = np.array(yvals)
yvals *= np.pi / 180.
xvals *= np.pi / 180.
# now make the 4 tests
for ys in yvals:
    starts = [[dot.switch, True],
              [hc.camera.clear_headings]]
    middles = [[hc.camera.import_config],
               [hc.camera.get_background, hc.window.get_frame],
               [dot.set_rz, xvals],
               [dot.set_rx, yvals]]
    ends = [[dot.switch, False],
            [hc.camera.reset_display_headings]]
    hc.scheduler.add_test(num_frames, starts, middles, ends)