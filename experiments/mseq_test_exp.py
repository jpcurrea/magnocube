"""Pat Shoemaker's stimulus for testing facilitation.

Designate an area of  60° x 60° in the receptive field of one eye, and define four parallel tracks with 20° separation between them. Under the assumption that LC11 is used in the detection of walking flies, maybe these should be horizontal tracks and the direction of motion should be away from the midline. Define a set of segment lengths for the different components of the experiment, from the full 60° length down to some value smaller than the LC11 receptive field – say, 60°, 30°, 20°, and 10°?  We’ll have to program those sequences appropriately – in the dragonfly experiments we specified that each jump between segments must move at least two tracks away laterally, and I think the starting points of the segments were randomized in the direction of motion (I wrote the Python script to generate the stimulus, but I’m barely literate in Python and I don’t remember exactly what I did 12 years ago!) I’m not sure that both constraints can be met with only 4 tracks, but we can work that out. During the experiment, each component is run with uninterrupted motion (i.e., never any delays between segments), but we imposed a significant rest period between different components to prevent habituation.

"""
import numpy as np
from math import ceil
import holocube.hc as hc
from holocube.camera import Camera


# calculate what the radius must be to form a 5 degree spot at a distance of 1
spot_rad = np.tan(5. * np.pi / 180. / 2)
spot = hc.stim.Sphere(window=hc.window, radius=spot_rad)

# Define the tracks
max_tracks = 5
track_width = 60   # degrees; represents the horizontal range of motion
track_height = 80    # degrees; the vertical range of tracks
# allow for moving the center of the tracks
center_x, center_y = -55, 0    # degrees



# we want the spot to rotate at ~200 degrees per second
# and it should traverse 60 degrees, so we need to know how many frames
speed = 200  # degrees per second
# speed /= 5
duration = track_width / speed
num_frames = int(duration * 60)  # 60 fps
# make sure that num_frames is divisible by max_tracks
num_frames += max_tracks - num_frames % max_tracks
# find the next value above num_frames that is divisible by 2, 3, and 4
factor = 2*3*2*5
num_frames = factor * ceil(num_frames / float(factor))

xvals = np.linspace(-track_width / 2, track_width / 2, num_frames)
ytracks = np.linspace(-track_height / 2, track_height / 2, max_tracks, endpoint=True)

order = [0, 2, 3, 1]

yvals = []
for num_tracks in range(max_tracks):
    num_tracks += 1
    # generate the order based on random selection of the available tracks,
    # avoiding tracks that are one away from the current track or more than 3 away
    # repeat this num_tracks * max_tracks times and then repeat each track for num_frames // num_tracks
    inds = np.arange(max_tracks)
    # iterate through each component (num_tracks) and select a random track that is not one away from the previous
    # there should be max_track components, each starting with a different track
    # within each of those components should be num_tracks tracks
    # selecting those tracks randomly but avoiding tracks that are one away from the previous track
    # and not more than 3 away
    order = np.zeros((max_tracks, num_tracks), dtype=int)
    order.fill(-1)
    last_val = -1
    # keep a list of potential starting vals
    start_vals = np.random.choice(inds, max_tracks, replace=False)
    # randomly select the start_vals such that there are no two that are one away from each other
    # and no two that are more than 3 away from each other
    start_vals = []
    last_val = -1
    temp_inds = np.copy(inds).tolist()
    for i in range(max_tracks):
        temp_arr = np.array(temp_inds)
        valid = np.ones_like(temp_inds, dtype=bool)
        if last_val > -1:
            valid = np.abs(temp_arr - last_val) > 1
        valid = np.logical_and(valid, np.abs(temp_arr - last_val) < 4)
        choice = np.random.choice(temp_arr[valid])
        start_vals.append(choice)
        last_val = choice
        temp_inds.remove(choice)
    start_vals = np.array(start_vals)
    order[:, 0] = start_vals
    # each component should start with one of the randomly selected start_vals
    # then, for each compent, we need to select from the inds that are left, randomly
    # but avoiding values within 1 of the last value and more than 3 away

    # we want to go in the order of the final sequence, storing the last value
    # and then selecting from the remaining values

    # if this is the beginning of the sequence, then we want to select randomly from the start_vals
    # otherwise, we need to select randomly from the remaining values

    for component, start_val in zip(range(max_tracks), start_vals):
        temp_inds = np.copy(inds).tolist()
        temp_inds.remove(start_val)
        for sub_track in range(1, num_tracks):
            # get the next val. if defined, then we want to select randomly from the remaining values
            next_ind = component * num_tracks + sub_track + 1
            next_component = next_ind // num_tracks
            next_sub_track = next_ind % num_tracks
            if next_ind < max_tracks * num_tracks:
                next_val = order[next_component, next_sub_track]
            else:
                next_val = -1
            # get the last val
            last_ind = component * num_tracks + sub_track - 1
            last_component = last_ind // num_tracks
            last_sub_track = last_ind % num_tracks
            last_val = order[last_component, last_sub_track]
            # we want to select randomly from the remaining values
            vals = temp_inds
            vals_arr = np.array(vals)
            # we want to select values that are more than 1 but less than 4 away from the last value
            valid = np.ones_like(vals_arr, dtype=bool)
            if last_val > -1:
                valid = np.abs(vals_arr - last_val) > 1
            valid = np.logical_and(valid, np.abs(vals_arr - last_val) < 4)
            # if next_val > 0:
            #     # avoid values within 1 of the next value
            #     valid = np.logical_and(valid, np.abs(vals_arr - next_val) != 0)
            # select
            choice = np.random.choice(vals_arr[valid])
            order[component, sub_track] = choice
            # remove this choice from the vals
            vals.remove(choice)
    # I found an error resulting in sequential values being selected
    # setup a break point to check the values
    if np.any(abs(np.diff(order, axis=1)) == 1):
        breakpoint()
    ys = ytracks[order]
    # now repeat and add the yvals so that there are num_frames of yvals
    yvals += [np.repeat(ys, num_frames // num_tracks, axis=1)]

yvals = np.array(yvals)
yvals -= center_y
xvals -= center_x


# # plot the 4 different tests
# from matplotlib import pyplot as plt
# fig, rows = plt.subplots(num_tracks, max_tracks, figsize=(max_tracks*2, num_tracks*2), sharex=True, sharey=True)
# # xs = np.tile(xvals, max_tracks)
# xs = xvals
# for row_num, (row, yval) in enumerate(zip(rows, yvals)):
#     for col_num, (ax, ys) in enumerate(zip(row, yval)):
#         # add thin gray lines for all of the tracks
#         for y in ytracks:
#             ax.axhline(y, color='gray', lw=0.5)
#         # and plot the yvals, splitting them into different lines with a square at the start and arrow at the end of 
#         # each line
#         y_set = np.unique(ys)
#         for y in y_set:
#             inds = ys == y
#             ax.plot(xs[inds], ys[inds], 'k')
#             ax.plot(xs[inds][0], y, 'ks')
#             ax.plot(xs[inds][-1], y, 'k>')
#         # for the first row, title the subplots with the test number
#         if row_num == 0:
#             ax.set_title(f'Test {col_num + 1}')
#         # for the first column, label the y-axis with the number of tracks
#         if col_num == 0:
#             if row_num == 0:
#                 ax.set_ylabel(f'{row_num + 1} track')
#             else:
#                 ax.set_ylabel(f'{row_num + 1} tracks')
#         ax.set_aspect('equal')
# plt.tight_layout()
# plt.show()

# breakpoint()
exp_starts = [[hc.window.set_bg, [.5, .5, .5, 1.]],
              [hc.window.set_far, 5],
              [hc.window.set_ref, 0, (0, 0, 0)],
              [hc.camera.clear_headings]]
exp_ends = [[hc.window.set_bg, [0., 0., 0., 1.]],
            [hc.window.set_ref, 0, (0, 0, 0)],
            [hc.camera.clear_headings]]
hc.scheduler.add_exp(name='mseq_test', starts=exp_starts, ends=exp_ends)

# get an msequence to set the ref colors
mseq = hc.tools.mseq(2, 8, 0, 0)
num_frames = len(mseq)
colors = np.zeros((num_frames, 3), dtype=int)
# when mseq == 1, set the ref color to white
colors[mseq == 1][:] = 255
# when mseq == -1, leave the ref color black

# elev_diff = np.append([0], np.diff(elev))
starts = [[spot.switch, True],
        [hc.window.set_ref, 0, (0, 0, 0)],
        [hc.camera.clear_headings]]
middles = [[hc.camera.import_config],
        [hc.camera.get_background, hc.window.get_frame],
        [hc.window.set_ref, 0, colors]]
ends = [[spot.switch, False],
        [spot.reset_pos_rot],
        [hc.window.set_ref, 0, (0, 0, 0)],
        [hc.window.reset_pos_rot],
        [hc.camera.reset_display_headings]]
hc.scheduler.add_test(num_frames, starts, middles, ends)