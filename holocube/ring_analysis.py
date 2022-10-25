"""An object for loading and processing fly heading in a magnetic tether.
"""
# for profiling
import cProfile
import pstats
from pstats import SortKey
# for general analysis
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats, ndimage
import skimage
import scipy
import sys
# for loading and saving video files
from skvideo import io
# for using the FileSelector dialog
from get_rings import *

blue, green, yellow, orange, red, purple = [
    (0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (0.83, 0.74, 0.37),
    (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]



class Track():
    def __init__(self, filename):
        """Load the video file and get metadata.


        Parameters
        ----------
        filename : str
            The path to the video file.
        """
        self.filename = filename
        self.subject = os.path.basename(filename).split(".")[0]
        # load video from matlab video
        if filename.endswith(".mat"):
            # self.video = io.vreader(self.filename)
            self.video = scipy.io.loadmat(self.filename)
            self.times = self.video['t_v'][:, 0]
            self.video = self.video['vidData'][:, :, 0] # height x width x channel x num_frames
            self.video = self.video.transpose(2, 0, 1)
        # or load a frame generator 
        else:
            self.video = io.vread(self.filename)
            breakpoint()
        # get video metadata
        self.num_frames, self.height, self.width = self.video.shape
        # get center and radii from the tracked file
        tracked_folder = "./tracking_data"
        tracked_fns = os.listdir(tracked_folder)
        tracked_fn = [os.path.join(tracked_folder, fn) for fn in tracked_fns if self.subject in fn][0]
        self.circle_data = np.load(tracked_fn)
        self.center = np.array([self.circle_data['x'][0], self.circle_data['y'][0]])
        self.small_radius, self.large_radius = self.circle_data[['radius_small', 'radius_large']][0]
        # get ring coordinates and angles
        self.set_rings(self.center, self.small_radius, self.large_radius)

    def get_background(self):
        """Get average frame of the whole video."""
        self.background = self.video.mean(-1)

    def set_rings(self, center, small_radius=10, large_radius=20, padding=3):
        """Define two rings for heading detection.


        Parameters
        ----------
        (center_x, center_y) : tuple, len=2
            The 2D coordinate of the center of both rings.
        small_radius : float, default=3
            The radius of the inner ring, which should intersect both sides 
            of the fly.
        large_radius : float, default=9
            The radius of the outer ring, which should intersect only the 
            abdomen.
        padding : int, default=3
            The padding the thicken the inter and outer rings.
        """
        x, y = center
        # make a mask of the two rings
        xs, ys = np.arange(self.width), np.arange(self.height)
        xgrid, ygrid = np.meshgrid(xs, ys)
        xgrid, ygrid = xgrid.astype(float), ygrid.astype(float)
        xgrid -= x
        ygrid -= y
        dists = np.sqrt(xgrid ** 2 + ygrid ** 2)
        angles = np.arctan2(ygrid, xgrid)
        # get indices of the two ring masks
        # inner ring and angles:
        include_inner = (dists >= small_radius - padding) * (dists <= small_radius + padding)
        ys_inner, xs_inner = np.where(include_inner)
        self.small_ring_coords = np.array([xs_inner, ys_inner]).T
        self.small_ring_angles = angles[include_inner]
        # outer ring and angles:
        include_outer = (dists >= large_radius - padding) * (dists <= large_radius + padding)
        ys_outer, xs_outer = np.where(include_outer)
        self.large_ring_coords = np.array([xs_outer, ys_outer]).T
        self.large_ring_angles = angles[include_outer]

    def get_heading(self, floor=5, ceiling=np.inf):
        """Threshold the video and get the heading for each frame.


        Parameters
        ----------
        floor : int, default=5
            The pixel value lower bound for the inclusive filter
        ceiling : int, default=np.inf
            The pixel value upper bound for the inclusive filter
        """
        self.heading = []
        for frame in self.video:
            # 1. get angles of the outside ring corresponding to the fly's tail end
            xs, ys = self.large_ring_coords.T
            outer_vals = frame[self.large_ring_coords[:, 1],
                               self.large_ring_coords[:, 0]]
            tail = (outer_vals > floor) * (outer_vals <= ceiling)
            tail_angs = self.large_ring_angles[tail]
            # 2. get circular mean of tail
            tail_dir = scipy.stats.circmean(tail_angs, low=-np.pi, high=np.pi)
            head_dir = tail_dir + np.pi
            if head_dir > np.pi:
                head_dir -= 2 * np.pi
            
            # 3. get bounds of head angles, ignoring angles within +/- 60 degrees of the tail
            lower_bounds, upper_bounds = [head_dir - np.pi/2], [head_dir + np.pi/2]
            # wrap bounds if they go outside of [-pi, pi]
            if lower_bounds[0] < -np.pi:
                lb = np.copy(lower_bounds[0])
                lower_bounds[0] = -np.pi
                lower_bounds += [lb + 2 * np.pi]
                upper_bounds += [np.pi]
            elif upper_bounds[0] > np.pi:
                ub = np.copy(upper_bounds[0])
                upper_bounds[0] = np.pi
                upper_bounds += [ub - 2 * np.pi]
                lower_bounds += [-np.pi]
            # 4. get angles of the inside ring corresponding to both head and tail
            head_angs = []
            for lb, ub in zip(lower_bounds, upper_bounds):
                include = (self.small_ring_angles > lb) * (self.small_ring_angles < ub)
                if np.any(include):
                    xs, ys = self.small_ring_coords[include].T
                    inner_vals = frame[self.small_ring_coords[include][:, 1],
                                       self.small_ring_coords[include][:, 0]]
                    head = (inner_vals > floor) * (inner_vals <= ceiling)
                    head_angs += [self.small_ring_angles[include][head]]
            head_angs = np.concatenate(head_angs)
            # 5. grab the head angs within those bounds
            self.heading += [scipy.stats.circmean(head_angs, low=-np.pi, high=np.pi)]
        # convert to ndarray
        self.heading = np.array(self.heading)
        
    def video_preview(self, vid_fn=None, relative=False, marker_size=3):
        """Generate video with headings superimposed.


        Parameters
        ----------
        relative : bool, default=False
            Whether to generate the video after removing the fly's motion.
        marker_size : int, default=3
            The side length in pixels of the square marker used to indicate 
            the head.
        """
        if vid_fn is None:
            vid_fn = ".".join(self.filename.split(".")[:-1])
            vid_fn += "_heading.mp4"
        # get new video parameters
        vid_radius = int(1.25 * self.large_radius)
        vid_center = np.array([vid_radius, vid_radius])
        # get the indices of the center pixels
        ylow = max(round(self.center[1]) - vid_radius, 0) 
        yhigh = min(round(self.center[1]) + vid_radius, self.height)
        xlow = max(round(self.center[0]) - vid_radius, 0)
        xhigh = min(round(self.center[0]) + vid_radius, self.width)
        # get the sizes for reindexing
        height, width = yhigh - ylow, xhigh - xlow
        new_height, new_width = vid_radius * 2, vid_radius * 2
        ystart = round((new_height - height)/2)
        ystop = ystart + height
        xstart = round((new_width - width)/2)
        xstop = xstart + width
        # crop the video first, to save time
        cropped_video = np.copy(self.video[:, ylow : yhigh, xlow : xhigh])[..., np.newaxis]
        # make an empty frame for temporary storage
        # frame_centered = np.zeros((2 * vid_radius, 2 * vid_radius, 3), dtype=float)
        # open a video and start storing frames
        # new_vid = io.FFmpegWriter(vid_fn)
        # todo: store into a numpy array and use vwrite instead
        new_vid = np.zeros((self.num_frames, 2 * vid_radius, 2 * vid_radius, 3), dtype='uint8')
        # for each frame:
        # for frame, orientation in zip(cropped_video, self.heading):
        for num, (frame, orientation, frame_centered) in enumerate(zip(
                cropped_video, self.heading, new_vid)):
            # get the head position
            d_vector = np.array([np.cos(orientation), np.sin(orientation)])
            pos = np.round(vid_center + self.small_radius * d_vector).astype(int)
            # make a version of the frame with a red square centered at the mean orientation
            frame_centered[ystart:ystop, xstart:xstop] = frame
            # if specified, rotate the frame to get relative motion
            if relative:
                frame_centered = ndimage.rotate(
                    frame_centered, (orientation + np.pi/2) * 180 / np.pi,
                    reshape=False)
            # otherwise, draw a line indicating the heading
            else:
                rr, cc, val = skimage.draw.line_aa(vid_radius, vid_radius, pos[0], pos[1])
                # scale down the older values
                old_vals = frame_centered[cc, rr].astype(float)
                old_vals *= (1 - val)[..., np.newaxis]
                frame_centered[cc, rr] = old_vals.astype('uint8')
                frame_centered[cc, rr, 0] = val * 255
                # frame_centered[cc, rr, 1:] = 10
                frame_centered[pos[1] - marker_size : pos[1] + marker_size,
                               pos[0] - marker_size : pos[0] + marker_size] = [155, 0, 0]
            # store the new frame
            # new_vid.writeFrame(frame_centered)
            # clear the temporary frame
            # frame_centered.fill(0)
            print_progress(num, self.num_frames)
        # new_vid.close()
        # use vwrite to store the array as a video
        io.vwrite(vid_fn, new_vid)
        
        print(f"Heading preview saved in {vid_fn}.")

    def graph_preview(self, bins=np.linspace(-np.pi, np.pi, 361)):
        """A Space X Time graph of inner, outer, and inner-outer rings


        Parameters
        ----------
        bins : array-like, default=np.linspace(-np.pi, np.pi, 361)
            Define the bounds of bins used for flattening the ring values.
        """
        num_bins = len(bins) - 1
        # store one space X time graph per ring
        graphs = []
        # for each ring:
        for angles, coords in zip(
                [self.small_ring_angles, self.large_ring_angles],
                [self.small_ring_coords, self.large_ring_coords]):
            # make empty array for storing the final graph
            vals = np.zeros((self.num_frames, num_bins), dtype='uint8')
            # group the lists of angles by the defined bins
            bin_groups = np.digitize(angles, bins)
            # get the video pixel values within these bins
            in_ring = self.video[:, coords[:, 1], coords[:, 0]]
            # sort the pixel values using the bin group labels
            order = np.argsort(bin_groups)
            in_ring = in_ring[:, order]
            bin_groups = bin_groups[order]
            changes = np.diff(bin_groups) > 0
            changes = np.where(changes)[0]
            # get binned pixel values
            bin_vals = np.split(in_ring, changes, axis=1)
            bin_vals = np.array([bin_val.mean(1) for bin_val in bin_vals])
            # store
            graphs += [bin_vals]
        # now graph:
        fig, axes = plt.subplots(ncols=2, sharey=True, sharex=True)
        time_bins = np.append([0], self.times)
        # 1. the outer ring
        axes[0].pcolormesh(bins, time_bins, graphs[0].T)
        # axes[0].scatter(self.headings, self.times, color=red, marker='.')
        # line plot excluding steps > np.pi
        speed = np.append([0], np.diff(self.heading))
        headings = np.copy(self.heading)
        headings[abs(speed) > np.pi] = np.nan
        axes[0].invert_yaxis()
        axes[0].plot(headings, self.times, color=red)
        axes[0].set_title("Inner Ring")
        # 2. the inner ring
        axes[1].pcolormesh(bins, time_bins, graphs[1].T)
        axes[1].set_title("Outer Ring")
        axes[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                           ["-$\pi$", "-$\pi$/2", "0", "$\pi$/2", "$\pi$"])
        # format
        plt.tight_layout()
        plt.show()
        
        def print_progress(part, whole):
    prop = float(part) / float(whole)
    sys.stdout.write('\r')
    sys.stdout.write('[%-20s] %d%%' % ('=' * int(20 * prop), 100 * prop))
    sys.stdout.flush()


if __name__ == "__main__":
    # manually select a file
    file_UI = FileSelector()
    file_UI.close()
    fn = file_UI.files[0]
    # process the heading and make a video
    track = Track(fn)
    track.get_heading()
    track.graph_preview()
    track.video_preview()
else:
    fns = os.listdir("./")
    video_fns = [fn for fn in fns if fn.endswith(".mat")]
    tracked_fns = os.listdir("tracking_data")
    for num, fn in enumerate(video_fns):
        # check if there is tracking data on this video
        base = os.path.basename(fn).split(".")[0]
        if any([base in fn for fn in tracked_fns]):
            # track the heading
            try:
                track = Track(fn)
                track.get_heading()
                # store the unwrapped orientations
                new_fn = ".".join(fn.split(".")[:-1])
                new_fn += "_tracking.npy"
                # np.save(new_fn, np.unwrap(track.heading))
                track.video_preview()
            except:
                pass
            print_progress(num + 1, len(tracked_fns))
        
