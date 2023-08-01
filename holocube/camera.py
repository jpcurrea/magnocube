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

try:
    import PySpin
    pyspin_loaded = True
except:
    pyspin_loaded = False

import scipy
from skvideo import io
import struct
import subprocess
import sys
import threading
import time
from queue import Queue
from collections import deque
import pickle
from PIL import Image

from matplotlib import pyplot as plt

np.int = np.int64
np.float = np.float64

if pyspin_loaded:
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()
    # Get current library version
    version = system.GetLibraryVersion()
    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
else:
    cam_list = [None]


# a class to transform a given image given the 4 corners of the destination
class TiltedPanel():
    def __init__(self, position=0, min_width=100, mirror_x=False, aspect_ratio=.5, img_shape=None):
        """Render an image as one of the panes in a simulated 3D cube.
        
        Parameters
        ----------
        position : int, default=0
            The position of the wall in the camera view. 0-3 correspond to
            bottom, right, top, and left respectively.
        min_width : int, default=100
            The resulting width of the smaller side of the panel, which will
            be adjacent to the original image. Normally, this will be equal to
            the side length of the image.
        mirror_x : bool, default=False
            Whether to flip the x-axis before doing the perspecive projection.
        aspect_ratio : float, default=.5
            The ratio of height to width before rotation.
        """
        pos_conv = np.array(['left', 'front', 'right', 'back'])
        if isinstance(position, str):
            ind = np.where(pos_conv == position)
            position = ind[0][0]
        self.position = position
        # convert string position to 0-3, 0=left, 1=front, 2=right, 3=back
        self.min_width = min_width
        self.mirror_x = mirror_x
        self.aspect_ratio = aspect_ratio
        self.img_shape = img_shape
        self.get_coords()

    def get_coords(self):
        """Prepare the coordinates needed for the PIL perspective projection."""
        # the input coords are the 4 corners of the original square image
        width = self.min_width
        height = width
        if self.img_shape is not None:
            height, width = self.img_shape[:2]
        self.input_coords = np.array([(0, 0), (width, 0), (width, height), (0, height)]).astype(int)
        # the output coordinates are the 4 corners of the resulting wedge
        length = 2 * self.aspect_ratio
        self.output_coords = np.array([[-1, 0], [1, 0], [length + 1, length], [-(length + 1), length]], dtype=float)
        self.output_coords *= float(self.min_width) / 2.
        # flip the x values if specified
        if self.mirror_x:
            self.output_coords[:, 0] *= -1
        # rotate the output coords depending on the given position
        self.output_coords -= self.output_coords.mean(0)
        # self.output_coords = np.rot90(self.output_coords, k=self.position)
        if self.position > 0:
            self.output_coords = rotate(self.output_coords, angle=self.position * np.pi/2)
        if self.position in [0, 2]:
            self.output_coords[:, 1] *= -1
        self.output_coords -= self.output_coords.min(0)
        self.output_coords = self.output_coords.astype(int)
        self.new_width, self.new_height = np.round(self.output_coords.max(0)).astype(int)
        # use algorithm to find the coefficients needed for the transform
        self.find_coeffs()

    def find_coeffs(self):
        pa, pb = self.output_coords, self.input_coords
        # print(pa, pb)
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        self.coeffs = np.array(res).reshape(8)        

    def project_image(self, arr):
        """Use PIL to render the image as if it were a panel orthogonal to the viewer.
        
        Parameters
        ----------
        arr : numpy.ndarray
            The image to transform as an ndarray. Assumes that the array has
            shape = (self.min_width, self.min_width)
        """
        # make a PIL.Image out of the input image
        if not isinstance(arr, Image.Image):
            # todo: this is taking the longest portion of the
            # program. if we can avoid converting to PIL, we can speed
            # this program up by a lot.
            self.image = Image.fromarray(arr)
        else:
            self.image = arr
        # perform the transform
        # todo: replace with numpy matrix multiplication
        new_img = self.image.transform(
            (self.new_width, self.new_height), Image.Transform.PERSPECTIVE,
            self.coeffs, Image.Resampling.BILINEAR)
        # # test: plot the original and transformed versions
        # fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
        # axes[0].imshow(np.array(self.image))
        # axes[1].imshow(np.array(new_img))
        # # get max size on each dimension
        # max_x, max_y = np.array([self.image.size, new_img.size]).max(0)
        # axes[1].set_xlim(max_x, 0)
        # axes[1].set_ylim(max_y, 0)
        # plt.show()
        # new_img.save(f"test_frame_{self.position}_{np.random.randint(0, 100000):5d}.png")
        return np.array(new_img)


def rotate(arr, angle=np.pi/2):
    """Use matrix multiplication to rotate the 2-D vectors by the specified amount.
    
    Parameters
    ----------
    arr : np.ndarray
        The vector of 2-D values to be rotated by the specified amount.
    angle : float, default=np.pi/2
        The angle to rotate the vectors.
    """
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle),  np.cos(angle)]])
    return np.dot(arr, rot_matrix)


def combine_panels(panels, imgs, output=None):
    """Combine 4 TiltedPanel objects into the border of a square image."""
    # make an empty imae if it wasn't provided
    if output is None:
        side_length = max([max(panel.new_width, panel.new_height) for panel in panels])
        output = np.zeros((side_length, side_length, 4), dtype='uint8')
    # add the projected panel values for each panel to the image
    for panel, img in zip(panels, imgs):
        height, width = panel.new_height, panel.new_width
        if panel.position in [0, 3]:
            output[:height, :width] += panel.project_image(img)
        else:
            output[-height:, -width:] += panel.project_image(img)
    # fill the diagonal lines with 0
    output[np.arange(side_length), -np.arange(side_length) - 1] = 0
    output[np.arange(side_length), np.arange(side_length)] = 0
    return output


class Kalman_Filter():
    '''
    2D Kalman filter, assuming constant acceleration.

    Use a 2D Kalman filter to return the estimated position of points given 
    linear prediction of position assuming (1) fixed jerk, (2) gaussian
    jerk noise, and (3) gaussian measurement noise.

    Parameters
    ----------
    num_objects : int
        Number of objects to model as well to expect from detector.
    sampling_interval : float
        Sampling interval in seconds. Should equal (frame rate) ** -1.
    jerk : float
        Jerk is modeled as normally distributed. This is the mean.
    jerk_std : float
        Jerk distribution standard deviation.
    measurement_noise_x : float
        Variance of the x component of measurement noise.
    measurement_noise_y : float
        Variance of the y component of measurement noise.

    '''
    def __init__(self, num_objects, num_frames=None, sampling_interval=30**-1,
                 jerk=0, jerk_std=125,
                 measurement_noise_x=5, measurement_noise_y=5,
                 width=None, height=None):
        self.width = width
        self.height = height
        self.num_objects = num_objects
        self.num_frames = num_frames
        self.sampling_interval = sampling_interval
        self.dt = self.sampling_interval
        self.jerk = jerk
        self.jerk_std = jerk_std
        self.measurement_noise_x = measurement_noise_x
        self.measurement_noise_y = measurement_noise_y
        self.tkn_x, self.tkn_y = self.measurement_noise_x, self.measurement_noise_y
        # process error covariance matrix
        self.Ez = np.array(
            [[self.tkn_x, 0         ],
             [0,          self.tkn_y]])
        # measurement error covariance matrix (constant jerk)
        self.Ex = np.array(
            [[self.dt**6/36, 0,             self.dt**5/12, 0,             self.dt**4/6, 0           ],
             [0,             self.dt**6/36, 0,             self.dt**5/12, 0,            self.dt**4/6],
             [self.dt**5/12, 0,             self.dt**4/4,  0,             self.dt**3/2, 0           ],
             [0,             self.dt**5/12, 0,             self.dt**4/4,  0,            self.dt**3/2],
             [self.dt**4/6,  0,             self.dt**3/2,  0,             self.dt**2,   0           ],
             [0,             self.dt**4/6,  0,             self.dt**3/2,  0,            self.dt**2  ]])
        self.Ex *= self.jerk_std**2
        # set initial position variance
        self.P = np.copy(self.Ex)
        ## define update equations in 2D as matrices - a physics based model for predicting
        # object motion
        ## we expect objects to be at:
        # [state update matrix (position + velocity)] + [input control (acceleration)]
        self.state_update_matrix = np.array(
            [[1, 0, self.dt, 0,       self.dt**2/2, 0           ],
             [0, 1, 0,       self.dt, 0,            self.dt**2/2],
             [0, 0, 1,       0,       self.dt,      0           ],
             [0, 0, 0,       1,       0,            self.dt     ],
             [0, 0, 0,       0,       1,            0           ],
             [0, 0, 0,       0,       0,            1           ]])
        self.control_matrix = np.array(
            [self.dt**3/6, self.dt**3/6, self.dt**2/2, self.dt**2/2, self.dt, self.dt])
        # measurement function to predict next measurement
        self.measurement_function = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]])
        self.A = self.state_update_matrix
        self.B = self.control_matrix
        self.C = self.measurement_function
        ## initialize result variables
        self.Q_local_measurement = []  # point detections
        ## initialize estimateion variables for two dimensions
        self.max_tracks = self.num_objects
        dimension = self.state_update_matrix.shape[0]
        self.Q_estimate = np.empty((dimension, self.max_tracks))
        self.Q_estimate.fill(np.nan)
        if self.num_frames is not None:
            self.Q_loc_estimateX = np.empty((self.num_frames, self.max_tracks))
            self.Q_loc_estimateX.fill(np.nan)
            self.Q_loc_estimateY = np.empty((self.num_frames, self.max_tracks))
            self.Q_loc_estimateY.fill(np.nan)
        else:
            self.Q_loc_estimateX = []
            self.Q_loc_estimateY = []
        self.num_tracks = self.num_objects
        self.num_detections = self.num_objects
        self.frame_num = 0

    def get_prediction(self):
        '''
        Get next predicted coordinates using current state and measurement information.

        Returns
        -------
        estimated points : ndarray
            approximated positions with shape.
        '''
        ## kalman filter
        # predict next state with last state and predicted motion
        self.Q_estimate = self.A @ self.Q_estimate + (self.B * self.jerk)[:, None]
        # predict next covariance
        self.P = self.A @ self.P @ self.A.T + self.Ex
        # Kalman Gain
        try:
            self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Ez)
            ## now assign the detections to estimated track positions
            # make the distance (cost) matrix between all pairs; rows = tracks and
            # cols = detections
            self.estimate_points = self.Q_estimate[:2, :self.num_tracks]
            # np.clip(self.estimate_points[0], -self.height/2, self.height/2, out=self.estimate_points[0])
            # np.clip(self.estimate_points[1], -self.width/2, self.width/2, out=self.estimate_points[1])
            return self.estimate_points.T  # shape should be (num_objects, 2)
        except:
            return np.array([np.nan, np.nan])
    

    def add_starting_points(self, points):
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        self.Q_estimate.fill(0)
        self.Q_estimate[:2] = points.T
        if self.num_frames is not None:
            self.Q_loc_estimateX[self.frame_num] = self.Q_estimate[0]
            self.Q_loc_estimateY[self.frame_num] = self.Q_estimate[1]
        else:
            self.Q_loc_estimateX.append(self.Q_estimate[0])
            self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1

    def add_measurement(self, points):
        ## detections matrix
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        self.Q_loc_meas = points
        # find nans, exclude from the distance matrix
        # no_nans_meas = np.isnan(self.Q_loc_meas[:, :self.num_tracks]) == False
        # no_nans_meas = no_nans_meas.max(1)
        # assigned_measurements = np.empty((self.num_tracks, 2))
        # assigned_measurements.fill(np.nan)
        # self.est_dist = scipy.spatial.distance_matrix(
        #     self.estimate_points.T,
        #     self.Q_loc_meas[:self.num_tracks][no_nans_meas])
        # use hungarian algorithm to find best pairings between estimations and measurements
        # if not np.any(np.isnan(self.est_dist)):
        # try:
        #     asgn = scipy.optimize.linear_sum_assignment(self.est_dist)
        # except:
        #     print(self.est_dist)
        # for num, val in zip(asgn[0], asgn[1]):
        #     assigned_measurements[num] = self.Q_loc_meas[no_nans_meas][val]
        # remove problematic cases
        # close_enough = self.est_dist[asgn] < 25
        # no_nans = np.logical_not(np.isnan(assigned_measurements)).max(1)
        # good_cases = np.logical_and(close_enough, no_nans)
        # if self.width is not None:
        #     in_bounds_x = np.logical_and(
        #         assigned_measurements.T[1] > 0, 
        #         assigned_measurements.T[1] < self.width)
        # if self.height is not None:
        #     in_bounds_y = np.logical_and(
        #         assigned_measurements.T[0] > 0, 
        #         assigned_measurements.T[0] < self.height)
        # good_cases = no_nans
        # good_cases = no_nans * in_bounds_x * in_bounds_y
        # apply assignemts to the update
        # for num, (good, val) in enumerate(zip(good_cases, assigned_measurements)):
        #     if good:
        #         self.Q_estimate[:, num] = self.Q_estimate[:, num] + self.K @ (
        #             val.T - self.C @ self.Q_estimate[:, num])
        #         self.track_strikes[num] = 0
        #     else:
        #         self.track_strikes[num] += 1
        #         self.Q_estimate[2:, num] = 0
        self.Q_estimate = self.Q_estimate + self.K @ (self.Q_loc_meas - self.C @ self.Q_estimate)
        # update covariance estimation
        self.P = (np.eye((self.K @ self.C).shape[0]) - self.K @ self.C) @ self.P
        ## store data
        if self.num_frames is not None:
            self.Q_loc_estimateX[self.frame_num] = self.Q_estimate[0]
            self.Q_loc_estimateY[self.frame_num] = self.Q_estimate[1]
        else:
            self.Q_loc_estimateX.append(self.Q_estimate[0])
            self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1

    def update_vals(self, **kwargs):
        """Allow for replacing parameters like jerk_std and noise estimates."""
        for key, val in kwargs.items():
            self.__setattr__(key, val)


class KalmanAngle(Kalman_Filter):
    """A special instance of the Kalman Filter for single object, 1D data."""
    def __init__(self, **kwargs):
        super().__init__(num_objects=1, measurement_noise_y=0, width=0, **kwargs) 
        self.last_point = 0
        self.revolutions = 0
        self.record = []
        # print key parameters
        print(f"jerk std={self.jerk_std}, noise std={self.measurement_noise_x}")

    def store(self, point):
        """Converts single point to appropriate shape for the 2D filter.
        
        Note: keep the point and last point variables in wrapped format and 
        unwrap just for when adding the measurement.
        """
        point = np.copy(point)
        if point is np.nan:
            point = self.last_point
        # unwrap
        if (self.last_point < -np.pi/2) and (point > np.pi/2):
            self.revolutions -= 1
        elif (self.last_point > np.pi/2) and (point < -np.pi/2):
            self.revolutions += 1
        self.last_point = np.copy(point)
        point += 2*np.pi*self.revolutions
        self.record += [point]
        # add 0 for second dimension
        point = np.array([point, 0])[np.newaxis]
        if self.frame_num == 0:
            self.add_starting_points(point)
        else:
            self.add_measurement(point)

    def predict(self):
        output = self.get_prediction()[0, 0]
        output %= 2*np.pi
        if output > np.pi:
            output -= 2*np.pi
        elif output < -np.pi:
            output += 2*np.pi
        return output


class Camera():
    def __init__(self, camera=cam_list[0], invert_rotations=True, kalman=True,
                 window=None, plot_stimulus=True, config_fn='video_player.config',
                 video_player_fn='video_player_server.py', com_correction=False):
        """Read from a BlackFly Camera via USB allowing GPIO triggers.


        Parameters
        ----------
        camera : PySpin.CameraBase
            The Camera instance from which to capture frames. Note: To run this 
            using a pre-recorded video, replace this with the path to the video.
        invert_rotations : bool, default=True
            Whether to apply a 90 degree rotation of the video before
            displaying.
        kalman : bool, default=True
            Whether to apply a Kalman filter to the heading data.
        window : holocube.window, default=None
            Option to provide a window instance for grabbing stimulus parameters
        plot_stimulus : bool, default=True
            Whether to add the stimulus to the border of the display.
        config_fn : str, default='video_player.config'
            Path to the config file used for configuring the video display.
        """
        # import the camera and setup the GenICam nodemap for PySpin
        self.com_correction = com_correction
        self.com_shift = None
        self.kalman = kalman
        self.camera = camera
        self.window = window
        self.background = None
        self.config_fn = config_fn
        self.video_player_fn = video_player_fn
        self.panels = []
        # track whether the camera is currently capturing, storing, and playing
        self.capturing = False
        self.storing = False
        self.playing = False
        self.dummy = False
        if isinstance(self.camera, str):
            self.dummy_fn = copy.copy(self.camera)
            self.dummy = True
            self.camera = None
            # load the video as a numpy array
            # self.video = io.vread(self.dummy_fn, as_grey=True)
            # self.video = io.FFmpegReader(self.dummy_fn)
            input_dict = {'-hwaccel': 'cuda', '-hwaccel_output_format': 'cuda'}
            self.video = io.FFmpegReader(self.dummy_fn, inputdict=input_dict)
            self.framerate = float(self.video.inputfps)
            self.save_fn = None
        elif pyspin_loaded:
            if isinstance(self.camera, PySpin.PySpin.CameraPtr):
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
        if self.camera is not None or self.dummy:
            # if the camera is loaded 
            self.get_video_params()
            # get camera information
            self.camera_info()
            # load the ring analysis parameters
            self.import_config()
            # setup the kalman filter
            if self.kalman:
                self.kalman_setup()
        # set default heading parameter
        self.heading = np.nan
        self.heading_smooth = np.nan
        self.north = 0
        self.offset = 0
        self.headings = []
        if self.kalman:
            self.headings_smooth = []
        else:
            self.headings_smooth = None
        # set whether to invert the rotations
        self.invert_rotations = invert_rotations
        self.gain = 1
        self.offset = 0

    def kalman_setup(self):
        self.kalman_filter = KalmanAngle(jerk_std=self.kalman_jerk_std, measurement_noise_x=self.kalman_noise)

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
        if pyspin_loaded:
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
            # get the framerate from the camera settings
            node_acquisition_framerate = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
            is_readable = (PySpin.IsAvailable(node_acquisition_framerate) and
                        PySpin.IsReadable(node_acquisition_framerate))
            assert is_readable, "Unable to get camera framerate"
            self.framerate = node_acquisition_framerate.GetValue()

    def get_video_params(self):
        """Grab important parameters of the video."""
        # get frame data by capturing one frame
        if self.camera is None:
            self.first_frame = self.video.__next__()
            if self.first_frame.ndim > 2:
                self.first_frame[..., 0]
        else:
            self.first_frame = self.grab_frames()[0]
        # check the frame shape
        self.shape = self.first_frame.shape
        # self.frame = mp.Array('i', self.first_frame.flatten())
        self.frame = np.copy(self.first_frame)
        self.height, self.width = self.first_frame.shape[:2]
        # use the inner and outer radii to get the ring indices
        self.img_xs = np.arange(self.width) - self.width/2
        self.img_ys = np.arange(self.height) - self.height/2
        self.img_xs, self.img_ys = np.meshgrid(self.img_xs, self.img_ys)
        # self.img_coords = np.array([self.img_xs, self.img_ys]).transpose(1, 2, 0)
        self.img_dists = np.sqrt(self.img_xs**2 + self.img_ys**2)
        self.img_angs = np.arctan2(self.img_ys, self.img_xs)

    def arm(self):
        """Arm the camera for frame acquisition."""
        # start acquiring frames
        self.camera.BeginAcquisition()

    def is_armed(self):
        if self.camera is None:
            return False
        else:
            return self.camera.IsStreaming()

    def disarm(self):
        """Disarm and release the camera."""
        # stop acquiring frames
        self.camera.EndAcquisition()

    def grab_frames(self, num_frames=1, timeout=1000):
        """Simply grab a number of frames from the camera."""
        if not self.dummy:
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
                # update frame number
                self.frame_num += 1

    def capture_dummy(self):
        self.frame_num = 0
        #self.frames = []
        self.num_frames = self.video.getShape()[0]
        while self.capturing:
            # grab frame from the video
            # self.frame = self.video[self.frame_num % len(self.video)]
            self.frame = self.video.__next__()
            if self.frame.ndim > 2:
                self.frame = self.frame[..., 0]
            self.update_heading()
            if self.frame_num > self.num_frames:
                self.clear_headings()
                self.video = io.FFmpegReader(self.dummy_fn)
                self.frame_num = 0
            self.frame_num += 1
            time.sleep(1./self.framerate)

    def capture_start(self):
        """Begin capturing, processing, and storing frames in a Thread."""
        if not self.dummy:
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
        self.import_config()
        if self.dummy:
            # start a thread to update the video frame
            self.capture_thread = threading.Thread(
                target=self.capture_dummy)
        else:
            # start a Thread to capture frames
            self.capture_thread = threading.Thread(
                target=self.capture)
        # start the thread
        self.capture_thread.start()

    def storing_start(self, duration=-1, dirname="./", save_fn=None,
                      capture_online=False):
        # todo: try replacing the vwrite method with an iterative ffmpeg writer
        self.frames = Queue()
        self.frame_num = 0
        self.frames_stored = 0
        self.headings = []
        self.headings_smooth = []
        self.heading_smooth = self.heading
        self.storing = True
        if not self.dummy and self.capturing:
            if save_fn is None:
                save_fn = f"{timestamp()}.mp4"
                save_fn = os.path.join(dirname, save_fn)
            self.save_fn = save_fn
            if capture_online:
                input_dict = {'-r': str(round(self.framerate)), '-hwaccel': 'cuda', '-hwaccel_output_format': 'cuda'}
                # output_dict = {'-r': str(round(self.framerate)), '-c:v': 'h264_nvenc', '-preset':'slow', '-vf':'hue=s=0'}
                # output_dict = {'-r': str(round(self.framerate)), '-c:v': 'h264_nvenc', '-preset':'p7', '-vf':'hue=s=0'}
                output_dict = {'-r': str(round(self.framerate)), '-c:v': 'h264_nvenc', '-preset':'lossless', '-vf':'hue=s=0'}
                self.video_writer = io.FFmpegWriter(save_fn, inputdict=input_dict, outputdict=output_dict)
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
        print("start storing frames")
        while self.storing or self.frames_stored < self.frame_num:
            frame = self.frames.get().reshape(self.height, self.width)
            frame = frame.astype('uint8')
            self.video_writer.writeFrame(frame)
            self.frames_stored += 1
            if self.storing:
                time.sleep(.002)

    def storing_stop(self):
        self.storing = False
        total_frames = self.frame_num
        if 'storing_thread' in dir(self):
            while self.frames_stored < total_frames:
                print_progress(self.frames_stored, self.frame_num)
        if "save_fn" in dir(self) and self.capturing:
            if 'video_writer' in dir(self):
                self.video_writer.close()
                print(f"{self.frames_stored} frames saved in {self.save_fn}")
            else:
                self.save_frames(self.save_fn)
                # self.frames = np.array(self.frames, dtype=)
                # io.vwrite(self.save_fn, self.frames,
                #     inputdict = {'-r': str(int(round(self.framerate)))},
                #     outputdict = {'-r': str(int(round(self.framerate)))})
                self.frames_stored = len(self.frames)
                # self.storing_thread = threading.Thread(target=self.store_frames)
                # self.storing_thread.start()
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

    def display_start(self):
        """Start the video_player_server.py subprocess to run in background."""
        # start the player
        args = ["python", self.video_player_fn, "-h", str(self.height),
                "-w", str(self.height), "-config_fn", str(self.config_fn)]
        self.video_player = subprocess.Popen(args, stdin=subprocess.PIPE)
        self.playing = True
        # use a thread to update the frame and heading
        self.update_heading()
        if self.kalman:
            self.update_display(self.frame, self.headings, self.headings_smooth)
        else:
            self.update_display(self.frame, self.headings)
        self.frame_updater = threading.Thread(
            target=self.display)
        self.frame_updater.start()

    def display_stop(self):
        assert 'video_player' in dir(self), "No dislay to stop!"
        self.playing = False
        if self.video_player.poll() is None:
            self.video_player.kill()

    def display(self, framerate=480.):
        """Save frame and heading for video_player_server.py to display."""
        # note: there were problems due to stray threads continuously running
        interval = 1/framerate
        diff = self.width - self.height
        start = diff//2
        end = self.width - start
        self.display_width = self.height
        while self.playing:
            # crop the frame to be a centered square
            if self.frame.ndim == 1:
                frame_cropped = self.frame.reshape(self.height, self.width).T
            elif self.frame.ndim > 2:
                frame_cropped = self.frame[..., 0].T
            else:
                frame_cropped = np.copy(self.frame).T
            frame_cropped = frame_cropped[start:end, :end-start]
            # grab stimulus frame and wrap around the cropped frame
            offset = 0
            if self.background is not None:
                img = self.get_border()
                # print(img.shape)
                # todo: fix img shape
                # save the output
                # image = image.fromarray(img)
                # image.save(f"{self.frame_num:05d}.png")
                offset = min(self.panels[0].new_height, self.panels[0].new_width)
                # breakpoint()
                # todo: fix min_width to make the final border dimensions correctly
                img[offset:offset + self.height, offset:offset + self.height] = frame_cropped[..., np.newaxis]
                frame_cropped = img
            self.update_display(frame_cropped, self.headings, self.headings_smooth, self.com_shift)
            # np.save(self.buffer_fn, frame_cropped)
            time.sleep(interval)

    def update_display(self, img, headings, headings_smooth=None, com_shift=None):
        """Send images and heading data to the video server using stdin.PIPE."""
        data = {'img': img, 'headings': headings, 'headings_smooth': headings_smooth, 'com_shift': com_shift}
        data_bytes = pickle.dumps(data)
        self.video_player.stdin.flush()
        # first, send the length of the data_bytes for safe reading
        length_prefix = struct.pack('I', len(data_bytes))
        self.video_player.stdin.write(length_prefix)
        # then, send the data
        self.video_player.stdin.write(data_bytes)
        self.video_player.stdin.flush()
        # self.video_player.communicate(input=data_bytes)

    def get_background(self, img):
        if callable(img):
            img = img()
        background = img.get_image_data()
        height, width = background.height, background.width
        self.background = np.array(background.get_data(), dtype='uint8').reshape(height, width,4)
        # plt.imsave(f"{self.frame_num:05d}.png", self.background)

    def get_border(self):
        # grab the whole window
        img_window = self.background
        # make a list of all of the images
        imgs = []
        for num, viewport in enumerate(self.window.viewports):
            # get the pixels for the viewport
            left, bottom, width, height = viewport.coords[0], viewport.coords[1], viewport.coords[2], viewport.coords[3]
            right, top = left + width, bottom + height
            panel_img = img_window[bottom:top, left:right]
            # print(panel_img.shape)
            # img = Image.fromarray(panel_img)
            # img.save(f"{self.frame_num:05d}_{num:01d}.png")
            imgs += [panel_img]
            height, width = panel_img.shape[:2]
            # make a TiltedPanel object from this
            if len(self.panels) < 4:
                # mirror_x = viewport.name in ['front', 'back']
                mirror_x = False
                panel = TiltedPanel(position=viewport.name, mirror_x=mirror_x, aspect_ratio=.25, min_width=self.display_width, img_shape=panel_img.shape)
                self.panels += [panel]
        # make the border
        return combine_panels(self.panels, imgs)

    def reset_display_headings(self):
        if 'headings_fn' in dir(self):
            np.save(self.headings_fn, np.array([]))

    def save_frames(self, new_fn):
        """Save the captured frames as a new video."""
        self.frames = np.concatenate(self.frames)
        self.frames = self.frames.reshape((self.frame_num, self.height, self.width))
        input_dict = {'-r': str(round(self.framerate)), '-hwaccel': 'cuda', '-hwaccel_output_format': 'cuda'}
        output_dict = {'-r': str(round(self.framerate)), '-c:v': 'h264_nvenc', '-preset':'slow', '-vf':'hue=s=0'}
        io.vwrite(new_fn, self.frames, inputdict=input_dict, outputdict=output_dict)
        print(f"{self.frame_num} frames saved in {new_fn}")

    def import_config(self, ring_thickness=3):
        # get the stored ring radii, threshold, and whether to invert
        info = configparser.ConfigParser()
        info.read(self.config_fn)
        if 'video_parameters' in info.keys():
            self.vid_params = info['video_parameters']
            for key, dtype in zip(['thresh', 'inner_r', 'outer_r', 'invert', 'flipped', 'kalman_jerk_std', 'kalman_noise'],
                                    [int, float, float, bool, bool, float, float]):
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
            # update the kalman filter if values changed
            if 'kalman_filter' in dir(self):
                std_changed = self.kalman_filter.jerk_std != self.kalman_jerk_std
                noise_changed = self.kalman_filter.measurement_noise_x != self.kalman_noise
                if std_changed or noise_changed:
                    self.kalman_setup()

    def update_heading(self, absolute=False):
        if self.playing:
            try:
                self.import_config()
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
            # todo: use index numbers instead of a boolean array, because it's faster to translate
            outer_inds, outer_angs = np.where(self.outer_inds), self.outer_angs
            inner_inds, inner_angs = np.where(self.inner_inds), self.inner_angs
            # outer_inds, outer_angs = self.outer_inds, self.outer_angs
            # inner_inds, inner_angs = self.inner_inds, self.inner_angs
            frame = self.frame
            # 1. get outer ring, which should include just the tail
            if frame.ndim == 1:
                frame = frame.reshape(self.height, self.width)
            elif frame.ndim > 2:
                frame = frame[..., 0]
            # todo: adjust inner and outer inds using the center of mass
            if self.com_correction:
                if invert:
                    frame_mask = frame < thresh
                else:
                    frame_mask = frame > thresh
                # account for shifts in the center of mass
                com = scipy.ndimage.measurements.center_of_mass(frame_mask)
                diff = np.array(com) - np.array([self.height/2, self.width/2])
                # outer_inds += np.round(diff).astype(int)[:, np.newaxis]
                # inner_inds += np.round(diff).astype(int)[:, np.newaxis]
            # use the new adjusted outer and inner inds
            # self.com = None
            outer_ring = frame[outer_inds[0], outer_inds[1]]
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
                        inner_vals = frame[inner_inds[0], inner_inds[1]][include]
                        if self.invert:
                            head = inner_vals < thresh
                        else:
                            head = inner_vals > thresh
                        heading = scipy.stats.circmean(
                            inner_angs[include][head],
                            low=-np.pi, high=np.pi)
            if self.com_correction:
                # convert from heading angle to head position
                heading_pos = self.inner_r * np.array([np.sin(heading), np.cos(heading)])
                # calculate direction vector between the center of the fly and the head
                direction = heading_pos - diff
                heading = np.arctan2(direction[0], direction[1])
                # store the shift in the center of mass for plotting
                self.com_shift = np.round(-diff).astype(int)
            # center and wrap the heading
            heading -= np.pi/2
            if heading < -np.pi:
                heading += 2 * np.pi
            # store
            self.heading = copy.copy(heading)
            self.headings += [self.heading]
            if self.kalman:
                # use kalman filter, only if heading is not np.nan
                if not np.isnan(heading):
                    # apply the kalman filter to the headings and output both the 
                    self.kalman_filter.store(heading)
                self.heading_smooth = self.kalman_filter.predict()
                if self.heading_smooth is np.nan:
                    self.kalman_setup()
                self.headings_smooth += [self.heading_smooth]
                return self.heading_smooth
            else:
                return heading
        else:
            self.heading = 0
            return self.heading

    def update_north(self, north):
        self.north = north

    def get_headings(self):
        ret = np.copy(self.headings)
        self.headings = []
        return ret

    def get_headings_smooth(self):
        ret = np.copy(self.headings_smooth)
        self.headings_smooth = []
        return ret

    def clear_headings(self):
        self.headings = []
        self.headings_smooth = []


class TrackingTrial():
    def __init__(self, camera, window, dirname, kalman=False):
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
        kalman : bool, default=False
            Whether to apply a Kalman filter to the processed heading.
        """
        self.camera = camera
        self.window = window
        self.dirname = dirname
        self.kalman = kalman
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
        if self.camera.capturing:
            if save_fn is None:
                self.fn = os.path.join(self.dirname, timestamp() + ".h5")
            else:
                self.fn = os.path.join(self.dirname, save_fn)
            if not os.path.exists(self.fn):
                temp_file = h5py.File(self.fn, 'w')
                temp_file.close()
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
        if self.camera.capturing:
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

    def add_test_data(self, arr, info=None, is_test=True):
        """Update the list of test data and info and camera-based tracking.

        Parameters
        ----------
        arr : array-like or function
            The array of heading data to store per test or a function returning
            the same.
        info : dict, default=None
            The dictionary of parameters to store alongside the heading data.
            Each parameter can be either a single number or an array of numbers
            with length = len(arr).
        is_test : bool, default=True
            Whether this entry is a test (as opposed to a rest period).

        Attributes
        ----------
        yaws : list
            The list of yaws corresponding to the virtual heading of the fly. If
            the fly is in closed loop, yaw should not change. If in open loop,
            yaw should equal the fly's heading.
        """
        if self.camera.capturing:
            if callable(arr):
                arr = arr()
            self.yaws += [arr]
            self.headings += [self.camera.get_headings()]
            self.virtual_headings += [self.virtual_objects['fly_heading'].get_angles()]
            self.virtual_headings_test = []
            self.tests += [is_test]
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
        if self.camera.capturing:
            if callable(value):
                value = value()
            self.h5_file.attrs[key] = value

    def save(self):
        """Store the test data and information into the H5 dataset."""
        if self.camera.capturing:
            # store the two heading measurements
            if len(self.virtual_headings) > 0:
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
                if isinstance(vals[0], (list, np.ndarray, tuple)):
                    # get the maximum length array and pad the others
                    max_len = max([len(val) for val in vals])
                else:
                    max_len = 1
                # make a nans array and store the values
                sub_val = vals[0]
                while isinstance(sub_val, (list, tuple)):
                    sub_val = sub_val[0]
                if isinstance(sub_val, str):
                    str_length = max([len(val) for val in vals])
                    dtype = ('S', str_length)
                else:
                    dtype = type(sub_val)
                arr = np.empty((len(vals), max_len), dtype=dtype)
                if np.issubdtype(dtype, np.floating):
                    arr[:] = np.nan
                elif np.issubdtype(dtype, np.integer):
                    arr[:] = 0
                else:
                    arr[:] = None
                for num, test in enumerate(vals):
                    if isinstance(test, (list, np.ndarray, tuple)):
                        arr[num, :len(test)] = test
                    else:
                        arr[num] = test
                # store the values in the h5 dataset
                try:
                    self.h5_file.create_dataset(param, data=np.squeeze(arr))
                except:
                    breakpoint()
            # new_fn = self.fn.replace(".h5", "_fail.h5")
            # new_dataset = h5py.File(new_fn, 'w')
            # # store all of the datasets 
            # for key in self.h5_file.keys():
            #     self.h5_file.copy(self.h5_file[key], new_dataset[key])
            # # and attributes from self.h5_file to new_dataset
            # for key in self.h5_file.attrs.keys():
            #     vals = self.h5_file[key]
            #     new_dataset[key] = vals
            # # now try closing the old dataset and check if it exists
            self.h5_file.close()
            time.sleep(1)        
            if os.path.exists(self.fn):
                print(f"data file stored at {self.fn}")
            else:
                print(f"data failed to save {self.fn}")
                # # now try closing the copy and check if that still exists
                # new_dataset.clos()
                # time.sleep(1)
                # if os.path.exists(new_fn):
                #     print(f"data file successfully stored at {new_fn}")


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

    def reset_virtual_object_motion(self):
        for lbl, object in self.virtual_objects.items():
            object.clear_motion()

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
        try:
            if np.isnan(heading):
                # use last angle that is not nan
                past_no_nans = np.isnan(self.past_angles) == False
                if np.any(past_no_nans):
                    last_non_nan = np.where(past_no_nans)[0][-1]
                    self.virtual_angle = self.past_angles[last_non_nan]
                else:
                    self.virtual_angle = 0
        except:
            print(heading)
            breakpoint()
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
        self.frame_num = 0

    def clear_motion(self):
        if 'offsets' in dir(self):
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
        # todo: try passing an individual frame to the gui using the PIPE method
        time.sleep(.5)
        cam.capture_start(dummy=True, 
                          dummy_fn=".\\revolving_fbar_different_starts\\2023_04_27_15_19_59.mp4")
        cam.storing_start(dummy=True)
        cam.import_config()
        # cam.set_heading_gain(0)
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

