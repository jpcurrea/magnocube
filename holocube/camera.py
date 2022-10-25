"""read and trigger a blackfly camera

Uses the PySide library for interfacing the camera. For now, I could only
install PySide on Python3.8 due to an error in pip preventing me from installing
on more recent versions.

The camera object needs to have the following methods:
__init__ : to initialize the specific camera
arm : arm the camera to be ready for capturing frames
capture : grab a frame from the camera feed
set_ring_values : set the two radii for head tracking
get_heading : get the heading using the two concentric rings
"""
import configparser
import copy
from datetime import datetime
import h5py
import numpy as np
import os
import PySpin
import scipy
from skvideo import io
import subprocess
import sys
import threading
import time
from queue import Queue

# Retrieve singleton reference to system object
system = PySpin.System.GetInstance()
# Get current library version
version = system.GetLibraryVersion()
# Retrieve list of cameras from the system
cam_list = system.GetCameras()

class Camera():
    def __init__(self, camera=cam_list[0], invert_rotations=True):
        """Read from a BlackFly Camera via USB allowing GPIO triggers.

        Parameters
        ----------
        camera : PySide.CameraBase
            The Camera instance from which to capture frames.
        invert_rotations : bool
            Whether to apply a 90 degree rotation of the video before
            displaying.
        """
        # import the camera and setup the GenICam nodemap for PySpin
        self.camera = camera
        self.camera.Init()
        self.nodemap = self.camera.GetNodeMap()
        # set the acquisition mode to continuous
        acquisition_mode = PySpin.CEnumerationPtr(
            self.nodemap.GetNode('AcquisitionMode'))
        ## check if acuisition mode node is writable
        is_writable = (PySpin.IsAvailable(acquisition_mode) and
                       PySpin.IsWritable(acquisition_mode))
        assert is_writable, "Unable to write node acquisition mode. Aborting..."
        ## check if continuous mode node is readable
        node_continuous = acquisition_mode.GetEntryByName('Continuous')
        is_readable = (PySpin.IsAvailable(node_continuous) and
                       PySpin.IsReadable(node_continuous))
        assert is_readable, "Unable to read node for continuous mode. Aborting..."
        ## set the acquisition mode
        continuous_val = node_continuous.GetValue()
        acquisition_mode.SetIntValue(continuous_val)
        # print('Acquisition mode set to continuous...')
        # now arm the camera
        self.arm()
        # track whether the camera is currently capturing and storing
        self.capturing = False
        self.storing = False
        # get camera information
        self.camera_info()
        # setup for the ring analysis
        self.playing = False
        # load the ring analysis parameters
        self.set_ring_params()
        # set default heading parameter
        self.heading = np.nan
        self.north = 0
        self.offset = 0
        self.headings = []
        # set whether to invert the rotations
        self.invert_rotations = invert_rotations
        self.gain = 1
        self.offset = 0

    def close(self, timeout=100):
        if "save_thread" in dir(self):
            if self.save_thread.is_alive():
                self.save_thread.join(timeout)
        if "capturing" in dir(self):
            if self.capturing:
                self.capture_stop()
        if "video_player" in dir(self):
            self.display_stop()
        if self.is_armed():
            self.disarm()
        del self.camera

    def camera_info(self):
        """Use PySpin to get camera device information"""
        self.nodemap_device = self.camera.GetTLDeviceNodeMap()
        # get device information node
        self.device_info_node = PySpin.CCategoryPtr(
            self.nodemap_device.GetNode('DeviceInformation'))
        # assume that the device info is available and readable
        is_readable = (PySpin.IsAvailable(self.device_info_node)
                       and PySpin.IsReadable(self.device_info_node))
        assert is_readable, "Device control information not available"
        # print key information about the camera
        features = self.device_info_node.GetFeatures()
        # print("\nDevice Information:")
        # for feature in features:
        #     node_feature = PySpin.CValuePtr(feature)
        #     if PySpin.IsReadable(node_feature):
        #         print(f"{node_feature.GetName()}: {node_feature.ToString()}")
        # get frame data by capturing one frame
        self.first_frame = self.grab_frames()[0]
        self.shape = self.first_frame.shape
        # self.frame = mp.Array('i', self.first_frame.flatten())
        self.frame = np.copy(self.first_frame)
        self.height, self.width = self.first_frame.shape
        # use the inner and outer radii to get the ring indices
        self.img_xs = np.arange(self.width) - self.width/2
        self.img_ys = np.arange(self.height) - self.height/2
        self.img_xs, self.img_ys = np.meshgrid(self.img_xs, self.img_ys)
        # self.img_coords = np.array([self.img_xs, self.img_ys]).transpose(1, 2, 0)
        self.img_dists = np.sqrt(self.img_xs**2 + self.img_ys**2)
        self.img_angs = np.arctan2(self.img_ys, self.img_xs)
        # get the framerate from the camera settings
        node_acquisition_framerate = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
        is_readable = (PySpin.IsAvailable(node_acquisition_framerate) and
                       PySpin.IsReadable(node_acquisition_framerate))
        assert is_readable, "Unable to get camera framerate"
        self.framerate = node_acquisition_framerate.GetValue()

    def arm(self):
        """Arm the camera for frame acquisition."""
        # start acquiring frames
        self.camera.BeginAcquisition()

    def is_armed(self):
        return self.camera.IsStreaming()

    def disarm(self):
        """Disarm and release the camera."""
        # stop acquiring frames
        self.camera.EndAcquisition()

    def grab_frames(self, num_frames=1, timeout=1000):
        """Simply grab a number of frames from the camera."""
        assert self.is_armed(), ("The camera is not armed. Try running "
                                 + "{self.arm} first.")
        # check if there are any capturing threads
        if self.capturing:
            self.capture_stop()
        # make an empty buffer to store all of the frames
        # self.buffer = np.zeros((num_frames, self.width, self.height))
        self.buffer = []
        for num in range(num_frames):
            # retrieve the latest
            frame = self.camera.GetNextImage(timeout).GetNDArray()
            self.buffer += [frame]
        # concatenate the buffer into an ndarray
        self.buffer = np.array(self.buffer)
        return self.buffer

    def capture_np(self, timeout=1000):
        """Simple loop for collecing frames using a numpy array buffer."""
        # until stop signal, capture frames in the buffer
        self.frame_num = 0
        self.frames = []
        while self.capturing:
            start = time.time()
            # retrieve the next frame
            frame = self.camera.GetNextImage(timeout).GetNDArray()
            # keep filling up the buffer until it's full, then copy the
            # frames to the frames list for stacking later
            self.buffer[self.frame_num % self.buffer.shape[0]][:] = frame
            self.get_heading(frame)
            # update the counter
            self.frame_num += 1
            if self.frame_num % self.buffer.shape[0] == 0:
                self.frames += [np.copy(self.buffer)]
            stop = time.time()
            print(stop - start)

    def capture(self, timeout=1000):
        """Simple loop for collecing frames. Alternative using list append."""
        # until stop signal, capture frames in the buffer
        self.frame_num = 0
        # setup optional display
        while self.capturing:
            # retrieve the next frame
            self.frame = self.camera.GetNextImage(timeout).GetData()
            # self.update_heading()
            if self.storing:
                # save the files
                # self.frames += [self.frame]
                if isinstance(self.frames, list):
                    self.frames += [self.frame]
                else:
                    self.frames.put(self.frame)
                # self.video_writer.writeFrame(
                #     self.frame.reshape(self.height, self.width))
                # self.headings += [self.heading]
                # update frame number
                self.frame_num += 1

    def capture_dummy(self):
        self.frame_num = 0
        #self.frames = []
        while self.capturing:
            # grab frame from the video
            self.frame = self.dummy_vid[self.frame_num % len(self.dummy_vid)]
            self.update_heading()
            if self.storing:
                # store frame
                # self.video_writer.writeFrame(
                #     self.frame.reshape(self.height, self.width))
                # self.frames += [self.frame]
                self.frames.put(self.frame)
                self.headings += [self.heading]
                # move onto the next frame
                self.frame_num += 1

    def capture_start(self, dummy=False, thread=True):
        """Begin capturing, processing, and storing frames in a Thread.

        Parameters
        ----------
        dummy : bool, default=False
            Whether to use a saved video for demo purposes.
        """
        # assume the camera is armed
        assert self.is_armed(), ("The camera is not armed. Try running "
                                 + "{self.arm} first.")
        # stop thread if still running and delete buffer
        if self.capturing:
            self.capture_stop()
        # we can stop the capture loop by setting self.capturing to False
        self.capturing = True
        self.storing = False
        # before capturing, update the ring parameters
        self.set_ring_params()
        if thread:
            if dummy:
                # load the video as a numpy array
                self.dummy_vid = io.vread("dummy_vid.mp4", as_grey=True)[..., 0]
                # start a thread to update the video frame
                self.capture_thread = threading.Thread(
                    target=self.capture_dummy)
            else:
                # toss current frame in case there are frames in the camera buffer
                # _ = self.grab_frames()
                # start a Thread to capture frames in parallel
                self.capture_thread = threading.Thread(
                    target=self.capture)
            # start the thread
            self.capture_thread.start()

    def storing_start(self, duration=-1, dirname="./", save_fn=None,
                      capture_online=True):
        # todo: try replacing the vwrite method with an iterative ffmpeg writer
        self.frames = Queue()
        self.frame_num = 0
        self.frames_stored = 0
        self.headings = []
        self.storing = True
        if save_fn is None:
            save_fn = f"{timestamp()}.mp4"
            save_fn = os.path.join(dirname, save_fn)
        self.save_fn = save_fn
        if capture_online:
            self.video_writer = io.FFmpegWriter(
                save_fn, inputdict={'-r': str(self.framerate)},
                outputdict={'-r': str(self.framerate)})
            self.storing_thread = threading.Thread(target=self.store_frames)
            self.storing_thread.start()
        else:
            self.frames = []
        if duration > 0 and save_fn is not None:
            time.sleep(duration)
            self.storing_stop(save_fn=save_fn)

    def get_save_fn(self):
        return self.save_fn

    def store_frames(self):
        """A thread to send frames to the ffmpeg writer"""
        while self.storing or self.frames_stored < self.frame_num:
            frame = self.frames.get().reshape(self.height, self.width)
            frame = frame.astype('uint8')
            self.video_writer.writeFrame(frame)
            self.frames_stored += 1

    def storing_stop(self):
        self.storing = False
        if 'storing_thread' in dir(self):
            while self.frames_stored < self.frame_num:
                print_progress(self.frames_stored, self.frame_num)
        elif "save_fn" in dir(self):
            self.save_frames(self.save_fn)
            # self.frames = np.array(self.frames, dtype=)
            # io.vwrite(self.save_fn, self.frames,
            #     inputdict = {'-r': str(int(round(self.framerate)))},
            #     outputdict = {'-r': str(int(round(self.framerate)))})
            self.frames_stored = len(self.frames)
            # self.storing_thread = threading.Thread(target=self.store_frames)
            # self.storing_thread.start()
        if 'video_writer' in dir(self):
            self.video_writer.close()
            print(f"{self.frames_stored} frames saved in {self.save_fn}")
        # assume the camera has frames stored
        # assert len(self.frames) > 0, ("No frames stored! First try running "
        #                               +f"{self.storing_start}")
        # if save_fn is not None:
        #     # make a Thread to save the files in parallel
        #     self.save_thread = threading.Thread(
        #         target=self.save_frames, kwargs={'new_fn': save_fn})
        #     self.save_thread.start()

    def capture_stop(self, save_fn=None):
        """Stop capturing frames and store the file."""
        # assume the camera currently capturing frames
        assert self.capturing, ("No capture thread to stop! First try running "
                                +f"{self.capture_start}")
        # stop the thread by setting the signal, self.capturing, to False
        self.capturing = False
        self.capture_thread.join(1000)
        # if the thread is still alive, there's a problem
        assert not self.capture_thread.is_alive(), "There was a problem closing the capture thread"

    def display_start(self, buffer_fn="_buffer.npy", heading_fn="_heading.npy",
                      north_fn="_north.npy"):
        """Start the video_player_server.py subprocess to run in background."""
        self.buffer_fn = buffer_fn
        self.heading_fn = heading_fn
        self.north_fn = north_fn
        # save the first frame
        self.display()
        # start the player
        args = ["python", "./video_player_server.py", "-h", str(self.height),
                "-w", str(self.width)]
        self.video_player = subprocess.Popen(args)
        self.playing = True
        # use a thread to update the frame and heading
        self.frame_updater = threading.Thread(
            target=self.display)
        self.frame_updater.start()

    def display_stop(self):
        assert 'video_player' in dir(self), "No dislay to stop!"
        if self.video_player.poll() is None:
            self.video_player.kill()

    def display(self, framerate=15.):
        """Save frame and heading for video_player_server.py to display."""
        # note: there were problems due to stray threads continuously running
        while self.playing:
            # self.update_heading()
            # arr = np.array([self.heading, self.north])
            # np.save(self.heading_fn, arr)
            np.save(self.heading_fn, self.heading)
            # store the latest frame for the video player
            np.save(self.buffer_fn, self.frame.reshape(self.height, self.width).T)
            time.sleep(1/framerate)

    def save_frames(self, new_fn):
        """Save the captured frames as a new video."""
        self.frames = np.concatenate(self.frames)
        self.frames = self.frames.reshape((self.frame_num, self.height, self.width))
        io.vwrite(new_fn, self.frames,
                  inputdict = {'-r': str(int(round(self.framerate)))},
                  outputdict = {'-r': str(int(round(self.framerate)))})
        print(f"{self.frame_num} frames saved in {new_fn}")
        # save the headings in a numpy array
        # heading_fn = ".".join(new_fn.split(".")[:-1])
        # heading_fn += "_headings.npy"
        # np.save(heading_fn, self.headings)
        # print(f"and {self.frame_num} headings saved in {heading_fn}")

    def set_ring_params(self, ring_thickness=3):
        try:
            # get the stored ring radii, threshold, and whether to invert
            info = configparser.ConfigParser()
            info.read("video_player.config")
            self.vid_params = info['video_parameters']
            for key, dtype in zip(['thresh', 'inner_r', 'outer_r', 'invert', 'flipped'],
                                  [int, float, float, bool, bool]):
                if dtype == bool:
                    val = self.vid_params[key] == 'True'
                else:
                    val = dtype(self.vid_params[key])
                setattr(self, key, val)
            # get the inner ring indices and angles
            self.inner_inds = self.img_dists <= (self.inner_r + ring_thickness/2.)
            self.inner_inds *= self.img_dists > (self.inner_r - ring_thickness/2.)
            # self.inner_inds = self.inner_inds.flatten()
            # self.inner_angs = self.img_angs.flatten()[self.inner_inds]
            self.inner_angs = self.img_angs[self.inner_inds]
            # get the outer ring indices and angles
            self.outer_inds = self.img_dists <= (self.outer_r + ring_thickness/2.)
            self.outer_inds *= self.img_dists > (self.outer_r - ring_thickness/2.)
            # self.outer_inds = self.outer_inds.flatten()
            # self.outer_angs = self.img_angs.flatten()[self.outer_inds]
            self.outer_angs = self.img_angs[self.outer_inds]
            # get experiment specific information
            self.exp_params = info['experiment_parameters']
            self.male = self.exp_params == 'male'
        except:
            pass

    def update_heading(self, absolute=False):
        try:
            self.set_ring_params()
        except:
            pass
        """Use stored radii to approximate the heading."""
        # 0. grab current value of pertinent variabls
        # thresh = np.copy(self.thresh)
        # invert = np.copy(self.invert)
        # outer_inds = np.copy(self.outer_inds)
        # outer_angs = np.copy(self.outer_angs)
        # inner_inds = np.copy(self.inner_inds)
        # inner_angs = np.copy(self.inner_angs)
        # frame = np.copy(self.frame)
        thresh, invert = self.thresh, self.invert
        outer_inds, outer_angs = self.outer_inds, self.outer_angs
        inner_inds, inner_angs = self.inner_inds, self.inner_angs
        frame = self.frame
        # 1. get outer ring, which should include just the tail
        if frame.ndim == 1:
            frame = frame.reshape(self.height, self.width)
        outer_ring = frame[outer_inds]
        heading = np.nan
        if self.flipped:
            # 2. if fly is forward leaning, then the head is in the outer ring
            # and we can ignore the inner ring
            if invert:
                head = outer_ring < thresh
            else:
                head = outer_ring > thresh
            if len(outer_angs) == len(head):
                head_angs = outer_angs[head]
                heading = scipy.stats.circmean(
                    head_angs.flatten(), low=-np.pi, high=np.pi)
        else:
            # 2. find the tail and head orientation by thresholding the outer ring
            # values and calculate the tail heading as the circular mean
            if invert:
                tail = outer_ring < thresh
            else:
                tail = outer_ring > thresh
            if len(outer_angs) == len(tail):
                tail_angs = outer_angs[tail]
                tail_dir = scipy.stats.circmean(tail_angs.flatten(), low=-np.pi, high=np.pi)
                # the head direction is the polar opposite of the tail
                head_dir = tail_dir + np.pi
                if head_dir > np.pi:
                    head_dir -= 2 * np.pi
                # 3. get bounds of head angles, ignoring angles within +/- 90 degrees of the tail
                lower_bounds, upper_bounds = [head_dir - np.pi / 2], [head_dir + np.pi / 2]
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
                lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)
                # 4. calculate the heading within the lower and upper bounds
                include = np.zeros(len(inner_angs), dtype=bool)
                for lower, upper in zip(lower_bounds, upper_bounds):
                    include += (inner_angs > lower) * (inner_angs < upper)
                if np.any(include):
                    if inner_inds.sum() == len(include):
                        inner_vals = frame[inner_inds][include]
                        if self.invert:
                            head = inner_vals < thresh
                        else:
                            head = inner_vals > thresh
                        heading = scipy.stats.circmean(
                            inner_angs[include][head],
                            low=-np.pi, high=np.pi)
        # invert the heading if specified
        # todo: why do I do this inversion? Is this the wrong place?
        # possible reasons: to match the orientation from the camera with that
        # of the arena in a common orientation
        # to match the camera orientation with that of the display
        # if the former, this inversion should be applied to the gain adjusted
        # value. If the latter, this inversion should happen in the display
        # function
        # todo: if I remove the inversion here, then the arena headings will be
        # inversely related to these?
        # if self.invert_rotations:
        #     heading *= -1
        # check if heading wrapped back around 0
        # if len(self.headings) > 1:
        #     last_heading = self.headings[-2]
        #     # two cases: clockwise and counterclockwise
        #     # if counterclockwise:
        #     if (last_heading > 3 * np.pi/4) and (heading < -3 * np.pi/4):
        #         self.rotation += 1
        #     # if clockwise:
        #     elif (last_heading < -3 * np.pi/4) and (heading > 3 * np.pi/4):
        #         self.rotation -= 1
        # # unwrap heading based on the number of rotations
        # heading += self.rotation * 2 * np.pi
        # # and return the gain and offset adjusted heading
        heading -= np.pi/2
        if heading < -np.pi:
            heading += 2 * np.pi
        # if self.invert_rotations:
        #     heading *= -1
        # rotate by pi/2 to match the projector coordinates
        # store
        self.heading = copy.copy(heading)
        self.headings += [self.heading]
        return heading
        # return self.gain * (heading - self.offset) + self.offset
        # return self.gain * (heading - self.offset)

    def update_north(self, north):
        self.north = north

    def get_headings(self):
        ret = np.copy(self.headings)
        self.headings = []
        return ret

    def clear_headings(self):
        self.headings = []

    # def set_heading_gain(self, gain, offset=None):
    #     if offset is not None:
    #         self.set_heading_offset(offset)
    #     else:
    #         if not np.isnan(self.heading):
    #             self.set_heading_offset(self.heading)
    #         else:
    #             # find the last heading that was not nan
    #             past_angles = self.headings[-1000:]
    #             no_nans = np.isnan(past_angles) == False
    #             if np.any(no_nans):
    #                 offset = past_angles[no_nans][-1]
    #                 self.set_heading_offset(offset)
    #             else:
    #                 self.set_heading_offset(0)
    #     self.gain = gain
    #
    # def set_heading_offset(self, offset):
    #     if callable(offset):
    #         offset = offset()
    #     offset %= 2 * np.pi
    #     self.offset = offset
    #     self.rotation = 0


class TrackingTrial():
    def __init__(self, camera, window, dirname):
        """Store the relevant variables into an h5 dataset.

        This object should help to connect the 1) the camera settings, like
        framerate, height, and width; 2) online tracking data from the camera;
        3) the online yaw data from the holocube display; and 4) any relevant
        DAQ data (to-do).

        Parameters
        ----------
        camera : Camera
            Camera instance (from this script) used to capture the fly
            heading.
        window : holocube.windows.Holocube_window
            Holocube_window instance from which to grab the relevant
            recorded data.
        dirname : str, path
            Path to the directory where to store the dataset.
        """
        self.camera = camera
        self.window = window
        self.dirname = dirname
        # make a filename for storing the dataset later
        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)
        # store a list of virtual objects to track
        self.virtual_objects = {}
        # setup a virtual fly heading object
        self.add_virtual_object(
            name='fly_heading', start_angle=self.camera.update_heading,
            motion_gain=-1, object=False)

    def h5_setup(self, save_fn=None):
        if save_fn is None:
            self.fn = os.path.join(self.dirname, timestamp() + ".h5")
        else:
            self.fn = os.path.join(self.dirname, save_fn)
        self.h5_file = h5py.File(self.fn, 'w')
        # make an empty list for test data
        self.tests = []
        self.yaws = []  # the list of yaw arrays per test from holocube
        self.headings = []  # the list of headings per test from the camera
        self.heading = 0
        self.virtual_headings_test = []  # the list of headings per test from the camera
        self.virtual_headings = []  # the list of headings per test from the camera
        self.test_info = {}  # the experiment parameters per test

    def store_camera_settings(self):
        """Store the camera settings for analysis later."""
        # self.camera.camera_info()
        info = {'framerate' : self.camera.framerate,
                'height' : self.camera.height,
                'width' : self.camera.width}
        # set camera info
        for key, item in info.items():
            self.h5_file.attrs[key] = item
        # set the ring GUI info
        for key, dtype in zip(['thresh', 'inner_r', 'outer_r', 'invert'],
                              [int, float, float, bool]):
            self.h5_file.attrs[key] = dtype(self.camera.vid_params[key])

    def set_test_params(self, params):
        """Set the list of parameter keys to store per test.

        Parameters
        ----------
        params : list
            The list of parameter labels to expect per test.
        """
        self.tests = []
        for param in params:
            # make empty lists to start appending
            self.test_info[param] = []

    def add_test_data(self, arr, info=None, rest=False):
        """Update the list of test data and info and camera-based tracking.

        Parameters
        ----------
        arr : array-like or function
            The array of heading data to store per test or a function returning
            the same.
        info : dict
            The dictionary of parameters to store alongside the heading data.
            Each parameter can be either a single number or an array of numbers
            with length = len(arr).

        Attributes
        ----------
        yaws : list
            The list of yaws corresponding to the virtual heading of the fly. If
            the fly is in closed loop, yaw should not change. If in open loop,
            yaw should equal the fly's heading.
        """
        if callable(arr):
            arr = arr()
        self.yaws += [arr]
        self.headings += [self.camera.get_headings()]
        self.virtual_headings += [self.virtual_objects['fly_heading'].get_angles()]
        self.virtual_headings_test = []
        self.tests += [rest]
        if info is not None:
            for param, value in info.items():
                if param not in self.test_info.keys():
                    self.test_info[param] = []
                if callable(value):
                    value = value()
                self.test_info[param] += [value]

    def add_exp_attr(self, key, value):
        """Update the list of test data and info and camera-based tracking.

        Parameters
        ----------
        data : dict
            The dictionary of experimental parameters to store alongside the
            dataset. Each parameter can be either a single number or an array of numbers
            with length = len(arr).
        """
        if callable(value):
            value = value()
        self.h5_file.attrs[key] = value

    def save(self):
        """Store the test data and information into the H5 dataset."""
        # store the two heading measurements
        max_len = max([len(test) for test in self.virtual_headings])
        new_virtual_headings = np.empty((len(self.virtual_headings), max_len), dtype=float)
        new_virtual_headings.fill(np.nan)
        for test, row in zip(self.virtual_headings, new_virtual_headings):
            row[:len(test)] = test
        self.virtual_headings = new_virtual_headings
        for vals, label in zip(
                [self.yaws, self.headings, self.virtual_headings],
                ['yaw', 'camera_heading', 'virtual_angle']):
            # remove any empty dimensions
            if len(vals) > 1:
                # convert the camera and display headings into 2D arrays
                # get the maximum length array and pad the others
                max_len = max([len(test) for test in vals])
                # make an nans array and store the values
                # dimensions: test X frame
                arr = np.empty((len(vals), max_len), dtype=float)
                arr[:] = np.nan
                if arr.size > 0:
                    for num, test in enumerate(vals):
                        arr[num, :len(test)] = test
            else:
                arr = np.array(vals)
            arr = np.squeeze(arr)
            # store the values in the h5 dataset
            self.h5_file.create_dataset(label, data=arr)
        # store indicator of whether the trial is a test or rest
        self.h5_file.create_dataset("is_test", data=np.array(self.tests))
        # store each test parameter as a dataset
        for param in self.test_info.keys():
            vals = self.test_info[param]
            # if elements are functions, replace with output
            functions = [callable(test) for test in vals]
            new_vals = []
            if any(functions):
                for test in vals:
                    if callable(test):
                        new_vals += [test()]
                    else:
                        new_vals += [test]
                vals = new_vals
            # if each val is an array
            try:
                if isinstance(vals[0], (list, np.ndarray, tuple)):
                    # get the maximum length array and pad the others
                    max_len = max([len(val) for val in vals])
                else:
                    max_len = 1
            except:
                breakpoint()
            # make a nans array and store the values
            arr = np.empty((len(vals), max_len), dtype=float)
            arr[:] = np.nan
            for num, test in enumerate(vals):
                if isinstance(test, (list, np.ndarray, tuple)):
                    try:
                        arr[num, :len(test)] = test
                    except:
                        breakpoint()
                else:
                    arr[num] = test
            # store the values in the h5 dataset
            self.h5_file.create_dataset(param, data=np.squeeze(arr))
        print(f"data file stored at {self.fn}")
        self.h5_file.close()

    def add_virtual_object(self, name, motion_gain=-1, start_angle=None,
                           object=True):
        # allow for passing functions as variables
        if callable(start_angle):
            start_angle = start_angle()
        if callable(motion_gain):
            motion_gain = motion_gain()
        # orientations with different gains
        self.position_gain = motion_gain + 1
        if start_angle is None:
            start_angle = self.heading
        # make a virtual object instance to keep track of other objects
        self.virtual_objects[name] = VirtualObject(
            start_angle, motion_gain, object, name)

    def update_objects(self, heading):
        self.heading = heading
        # self.virtual_objects['fly_heading'].update_angle(self.heading)
        for lbl, object in self.virtual_objects.items():
            object.update_angle(self.heading)

    def get_object_heading(self, lbl):
        """Get the fly heading and store for later."""
        return self.virtual_objects[lbl].get_angle()

    def record_timestamp(self):
        """Simply store the current timestamp for later."""
        self.timestamp = time.time()

    def get_timestamp(self):
        return self.timestamp

    # def get_virtual_heading(self):
    #     # the virtual heading should be scaled by the position gain
    #     # but in such a way that changing the gain doesn't cause motion
    #     self.heading = self.camera.update_heading()
    #     # determine if a revolution has been made
    #     if len(self.camera.headings) > 1:
    #         last_heading = self.camera.headings[-2]
    #         # two cases: clockwise and counterclockwise
    #         # if counterclockwise:
    #         if (last_heading > 3 * np.pi/4) and (self.heading < -3 * np.pi/4):
    #             self.revolution += 1
    #         # if clockwise:
    #         elif (last_heading < -3 * np.pi/4) and (self.heading > 3 * np.pi/4):
    #             self.revolution -= 1
    #     # unwrap the heading data based on the number of revolutions
    #     self.heading += self.revolution * 2 * np.pi
    #     # calculate the virtual heading
    #     self.virtual_angle = self.position_gain * (self.heading - self.start_angle) + self.start_angle
    #     self.virtual_headings_test += [self.virtual_angle]
    #     # get heading for each virtual object
    #     return self.virtual_angle
    #
    # def set_motion_parameters(self, motion_gain, start_angle=None):
    #     self.position_gain = motion_gain + 1
    #     if start_angle is None:
    #         start_angle = self.heading
    #     elif callable(start_angle):
    #         start_angle = start_angle()
    #     self.start_angle = start_angle
    #     self.revolution = 0
    #
    # def virtual_to_camera(self, heading=0):
    #     """Convert virtual coordinates to the corresponding camera coordinates.
    #
    #     heading = self.position_gain * (self.heading - self.start_angle) + self.start_angle
    #     => heading = (heading - self.start_angle)/self.position_gain + self.start_angle
    #
    #     Parameters
    #     ----------
    #     heading : float or array-like, default=0
    #     The heading or array of angles in virtual coordinates to convert.
    #     """
    #     if self.position_gain == 0:
    #         heading = self.start_angle
    #     else:
    #         heading = (heading - self.start_angle) / self.position_gain + self.start_angle
    #     return heading

class VirtualObject():
    def __init__(self, start_angle=0, motion_gain=-1, object=True, name=None):
        """Keep track of the angular location of a virtual object.

        Parameters
        ----------
        start_angle : float, default=0
            The original orientation of the object.
        motion_gain : float, default=-1
            The motion gain to apply to the heading transformation.
        object : bool, default=True
            Whether this is an external object or a viewing vector.
            An object orientation should have the inverse relationship
            with camera angles as a viewing vector.
        """
        self.set_motion_parameters(motion_gain, start_angle)
        self.virtual_angle = self.start_angle
        self.revolution = 0
        self.past_angles = []
        self.frame_num = 0
        self.object = object
        self.name = name

    def update_angle(self, heading):
        if np.isnan(heading):
            # use last angle that is not nan
            past_no_nans = np.isnan(self.past_angles) == False
            if np.any(past_no_nans):
                last_non_nan = np.where(past_no_nans)[0][-1]
                self.virtual_angle = self.past_angles[last_non_nan]
            else:
                self.virtual_angle = 0
        else:
            if len(self.past_angles) > 1:
                last_angle = self.past_angles[-1]
                # two cases: clockwise and counterclockwise
                # if counterclockwise:
                if (last_angle > 3 * np.pi / 4) and (heading < -3 * np.pi / 4):
                    self.revolution += 1
                # if clockwise:
                elif (last_angle < -3 * np.pi / 4) and (heading > 3 * np.pi / 4):
                    self.revolution -= 1
            # unwrap the heading data based on the number of revolutions
            heading += self.revolution * 2 * np.pi
            # calculate the virtual heading
            mod = -1
            if self.object:
                mod = 1
            self.virtual_angle = mod * self.position_gain * (
                    heading - self.start_angle) + self.start_angle
        # self.virtual_angle = mod * self.position_gain * heading
        # check for any predefined motion
        if 'offsets' in dir(self):
            offset = self.offsets[self.frame_num % len(self.offsets)]
            self.virtual_angle += offset
        self.past_angles += [self.virtual_angle]
        # update frame counter
        self.frame_num += 1

    def set_motion_parameters(self, motion_gain, start_angle=None):
        self.motion_gain = motion_gain
        self.position_gain = motion_gain + 1
        if start_angle is None:
            start_angle = self.virtual_angle
        elif callable(start_angle):
            start_angle = start_angle()
        self.start_angle = start_angle
        self.revolution = 0
        self.frame_num = 0

    def clear_angles(self):
        self.past_angles = []

    def get_angle(self):
        return self.virtual_angle

    def get_angles(self):
        ret = self.past_angles
        self.clear_angles()
        return ret

    def virtual_to_camera(self, virtual_angles):
        """Convert virtual coordinates to the corresponding camera coordinates.

        real angle = self.position_gain * (virtual angle - start angle) + start angle
        => heading = (virtual angle - start_angle)/position gain + start_angle

        Parameters
        ----------
        virtual_angles : float or array-like, default=0
            The heading or array of angles in virtual coordinates to convert.
        """
        if self.position_gain == 0:
            real_angle = self.start_angle
        else:
            real_angle = (virtual_angles - self.start_angle) / self.position_gain + self.start_angle
        return real_angle

    def add_motion(self, offsets):
        """Add motion to the virtual object.

        Parameters
        ----------
        offsets : float or array-like
            The value(s) to add to the virtual angle.
        """
        self.offsets = offsets

    def clear_motion(self):
        delattr(self, 'offsets')

def print_progress(part, whole):
    import sys
    prop = float(part) / float(whole)
    sys.stdout.write('\r')
    sys.stdout.write('[%-20s] %d%%' % ('=' * int(20 * prop), 100 * prop))
    sys.stdout.flush()

def timestamp():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")


if __name__ == "__main__":
    # start the QT loop
    # Get current library version
    version = system.GetLibraryVersion()
    print("PySpin version: "
          f"{version.major}.{version.minor}.{version.type}.{version.build}")
    # Retrieve list of cameras from the system
    num_cameras = cam_list.GetSize()
    # print('Number of cameras detected:', num_cameras)
    # Finish if there are no cameras
    if num_cameras > 0:
        # Use the first indexed camera
        cam = Camera(cam_list[0])
        # tracker = TrackingTrial(camera=cam, window=None, dirname="test")
        # tracker.set_test_params(['cond'])
        cam.display_start()
        time.sleep(.5)
        cam.capture_start(dummy=False)
        cam.storing_start()
        cam.set_ring_params()
        cam.set_heading_gain(0)
        wait = input("press any button to stop")
        # info = np.random.random(len(cam.headings))
        # tracker.add_test_data(info, {'cond': info})
        # tracker.save()
        cam.storing_stop(save_fn="test.mp4")
        cam.set_heading_gain(0)
        cam.close()
    else:
        print('No cameras!')
    # Clear camera list before releasing system
    cam_list.Clear()
    # Release instance
    system.ReleaseInstance()

