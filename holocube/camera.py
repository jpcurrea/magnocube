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
    cam_list = []

hq_video = "test.mp4"
# if True:
if len(cam_list) == 0:
    cam_list = [hq_video]
    pyspin_loaded = False

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
        self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Ez)
        ## now assign the detections to estimated track positions
        # make the distance (cost) matrix between all pairs; rows = tracks and
        # cols = detections
        self.estimate_points = self.Q_estimate[:2, :self.num_tracks]
        # np.clip(self.estimate_points[0], -self.height/2, self.height/2, out=self.estimate_points[0])
        # np.clip(self.estimate_points[1], -self.width/2, self.width/2, out=self.estimate_points[1])
        return self.estimate_points.T  # shape should be (num_objects, 2)
        # except:
        #     breakpoint()
        #     return np.array([np.nan, np.nan])
    

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
        # print(f"jerk std={self.jerk_std}, noise std={self.measurement_noise_x}")

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
            if 'K' not in dir(self):
                self.predict()
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
                 video_player_fn='video_player_server.py', com_correction=True,
                 saccade_trigger=True, acceleration_thresh=1000, wing_analysis=False):
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
        video_player_fn : str, default='video_player_server.py'
            Path to the video player server file.
        com_correction : bool, default=False
            Whether to apply a correction to the heading based on the center of
            mass of the stimulus.
        saccade_trigger : bool, default=True
            Whether to use the saccade trigger to start and stop recording.
        acceleration_thresh : float, default=1
            The threshold acceleration for detecting saccades.
        wing_analysis : bool, default=False
            Whether to use the wing analysis algorithm to detect saccades.
        """
        # import the camera and setup the GenICam nodemap for PySpin
        self.frame_num = 0
        self.frames_stored = 0
        self.buffer_start, self.buffer_stop = 0, 0
        self.buffer_ind = 0
        self.frames_to_process = []
        self.wing_analysis = wing_analysis
        self.saccade_trigger = saccade_trigger
        self.acceleration_thresh = acceleration_thresh
        self.speed_thresh = 0
        self.is_saccading = False
        self.com_correction = com_correction
        self.com_shift = [0, 0]
        self.kalman = kalman
        self.camera = camera
        self.window = window
        self.background = None
        self.config_fn = os.path.abspath(config_fn)
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
        if not self.dummy:
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
        print(f"fps = {self.framerate}")

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
        self.img_coords = np.array([self.img_xs, self.img_ys]).transpose(1, 2, 0)
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
        self.buffer_start, self.buffer_stop = 0, 0
        # todo: make a numpy array for storing frames between heading updates
        # the size of the buffer should be the ratio of the camera and the stimulus framerates
        # buffer_size = self.framerate / self.
        # self.frames_to_process = np.zeros()
        # setup optional display
        while self.capturing:
            # retrieve the next frame
            self.frame = self.camera.GetNextImage(timeout).GetData()
            # store 
            # assume the buffer is large enough to store the frames
            # if self.buffer_ind >= len(self.buffer):
            #     self.buffer_ind = 0
            # assert self.buffer_ind < len(self.buffer), "Buffer is too small to store frames at this framerate"
            # self.buffer[self.buffer_ind] = self.frame.reshape((self.height, self.width))
            # self.buffer_ind += 1
            # todo: convert to using a start and stop point in the buffer to function circularly
            self.buffer[self.buffer_stop] = self.frame.reshape((self.height, self.width))
            self.buffer_stop += 1
            self.buffer_stop %= len(self.buffer)
            # self.frame_num += 1
            # self.frames_to_process += [self.frame]
            # if self.storing:
            #     self.frames.put(self.frame)
            #     self.frame_num += 1
            # self.update_heading()
            # if self.storing:
            #     # save the files
            #     # self.frames += [self.frame]
            #     if isinstance(self.frames, list):
            #         self.frames += [self.frame]
            #     else:
            #         self.frames.put(self.frame)
                # update frame number

    def capture_dummy(self):
        #self.frames = []
        self.frame_num = 0
        self.vid_frame_num = 0
        self.num_frames = self.video.getShape()[0]
        self.framerate = 120
        while self.capturing:
            # grab frame from the video
            # self.frame = self.video[self.frame_num % len(self.video)]
            self.frame = self.video.__next__()
            if self.frame.ndim > 2:
                self.frame = self.frame[..., 0]
            # assert self.buffer_ind < len(self.buffer), "Buffer is too small to store frames at this framerate"
            # if self.buffer_ind >= len(self.buffer):
            #     self.buffer_ind = 0
            # self.buffer[self.buffer_ind] = self.frame
            self.buffer[self.buffer_stop % len(self.buffer)] = self.frame
            self.buffer_stop += 1
            self.buffer_stop %= len(self.buffer)
            self.vid_frame_num += 1
            # self.frames_to_process += [self.frame]
            if self.vid_frame_num % self.num_frames == 0:
                self.clear_headings()
                self.reset_display()
                self.reset_data()
                self.video = io.FFmpegReader(self.dummy_fn)
                self.vid_frame_num = 0
            time.sleep(1.0/self.framerate)
            if not self.storing and self.vid_frame_num % 4 == 0:
                self.update_heading()
                self.import_config()

    def capture_start(self, buffer_size=100):
        """Begin capturing, processing, and storing frames in a Thread."""
        self.frames = Queue()
        # make an array buffer to temporarily store frames
        self.buffer = np.zeros((buffer_size, self.height, self.width), dtype='uint8')
        # self.buffer_ind = 0
        self.buffer_start = 0
        self.buffer_stop = 0
        # self.img_coords = np.repeat(self.img_coords[np.newaxis], buffer_size, axis=0)
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
            # todo: grab 4 frames to test the update_heading function
            for rep in range(8):
                self.frame = self.video.__next__()
                if self.frame.ndim > 2:
                    self.frame = self.frame[..., 0]
                assert self.buffer_ind < len(self.buffer), "Buffer is too small to store frames at this framerate"
                self.buffer[self.buffer_stop] = self.frame
                self.buffer_ind += 1
            self.playing = True
            self.update_heading()
            # start a thread to update the video frame
            self.capture_thread = threading.Thread(
                target=self.capture_dummy)
        else:
            # start a Thread to capture frames
            self.capture_thread = threading.Thread(
                target=self.capture)
        # start the thread
        self.capture_thread.start()

    def storing_start(self, duration=-1, dirname="./", save_fn=None, capture_online=True):
        self.frames = Queue()
        self.frame_num = 0
        self.frames_stored = 0
        self.headings = []
        self.headings_smooth = []
        self.heading_smooth = self.heading
        self.storing = True
        # import the gui-set configuration before storing frames
        self.import_config()
        # if not self.dummy and self.capturing:
        if self.capturing:
            if save_fn is None:
                save_fn = f"{timestamp()}.mp4"
                save_fn = os.path.join(dirname, save_fn)
            self.save_fn = save_fn
            if capture_online:
                input_dict = {'-r': str(round(self.framerate)), '-hwaccel': 'cuda', '-hwaccel_output_format': 'cuda'}
                output_dict = {'-r': str(round(self.framerate)), '-c:v': 'h264_nvenc', '-preset':'slow', '-vf':'hue=s=0'}
                # output_dict = {'-r': str(round(self.framerate)), '-c:v': 'h264_nvenc', '-preset':'p7', '-vf':'hue=s=0'}   
                # output_dict = {'-r': str(round(self.framerate)), '-c:v': 'h264_nvenc', '-preset':'lossless', '-vf':'hue=s=0'}
                self.video_writer = io.FFmpegWriter(save_fn, inputdict=input_dict, outputdict=output_dict)
                self.storing_thread = threading.Thread(target=self.store_frames)
                self.storing_thread.start()
            else:
                self.frames = []
            if duration > 0 and save_fn is not None:
                time.sleep(duration)
                self.storing_stop(save_fn=save_fn)
        self.reset_display()
        self.reset_data()

    def get_save_fn(self):
        return self.save_fn

    def store_frames(self):
        """A thread to send frames to the ffmpeg writer"""
        print("start storing frames")
        while self.storing:
            self.store_frame()
            time.sleep(.002)

    def store_frame(self):
        """Store the next frame."""
        if self.frames_stored < self.frame_num:
            frame = self.frames.get().reshape(self.height, self.width)
            frame = frame.astype('uint8')
            self.video_writer.writeFrame(frame)
            self.frames_stored += 1

    def storing_stop(self):
        self.storing = False
        if 'storing_thread' in dir(self):
            cancel = False
            resp = input("press <c> to cancel recording or <enter> to continue: ")
            if resp == 'c':
                cancel = True
            if not cancel:
                while self.frames_stored < self.frame_num:
                    self.store_frame()
                    print_progress(self.frames_stored, self.frame_num)
#        if "save_fn" in dir(self) and not self.dummy:
        if "save_fn" in dir(self):
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
            if cancel:
                if self.frames_stored > 0:
                    self.video_writer.close()
                os.remove(self.save_fn)
        # elif 'video_writer' in dir(self) and cancel:
        #     try:
        #         self.video_writer.close()
        #     except:
        #         pass
            # os.remove(self.save_fn)
        # assume the camera has frames stored
        # assert len(self.frames) > 0, ("No frames stored! First try running "
        #                               +f"{self.storing_start}")
        # if save_fn is not None:
        #     # make a Thread to save the files in parallel
        #     self.save_thread = threading.Thread(
        #         target=self.save_frames, kwargs={'new_fn': save_fn})
        #     self.save_thread.start()
        self.reset_data()

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
        self.frames = Queue()
        self.frame_num = 0
        self.frames_stored = 0
        self.headings = []
        self.headings_smooth = []
        self.heading_smooth = self.heading
        # make a dictionary to keep track of data to be PIPEd to the video player
        self.display_data = {'heading': [], 'heading_smooth': [], 'com_shift': [], 
                             'wing_left': [], 'wing_right': [], 'head_angle': [],
                             'left_haltere': [], 'right_haltere': []}
        # start the player
        args = ["python", self.video_player_fn, "-h", str(self.height),
                "-w", str(self.height), "-config_fn", str(self.config_fn), 
                "-wingbeat", str(self.wing_analysis)]
        self.video_player = subprocess.Popen(args, stdin=subprocess.PIPE)
        self.playing = True
        # use a thread to update the frame and heading
        # self.update_heading()
        # self.signal_display(img=self.frame, **self.display_data)
        self.frame_updater = threading.Thread(
            target=self.display)
        self.frame_updater.start()

    def display_stop(self):
        assert 'video_player' in dir(self), "No dislay to stop!"
        self.playing = False
        if self.video_player.poll() is None:
            self.video_player.kill()

    def display(self, framerate=60.):
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
            # invert vals if specified
            # if self.invert:
            #     frame_cropped[..., :3] = 255 - frame_cropped[..., :3]
            # grab stimulus frame and wrap around the cropped frame
            offset = 0
            if self.background is not None:
                img = self.get_border()
                offset = min(self.panels[0].new_height, self.panels[0].new_width)
                # fix min_width to make the final border dimensions correctly
                img[offset:offset + self.height, offset:offset + self.height] = frame_cropped[..., np.newaxis]
                frame_cropped = img
            # self.signal_display(img=frame_cropped, heading=self.heading, heading_smooth=self.heading_smooth, com_shift=self.com_shift)
            # print(self.display_data)
            self.signal_display(img=frame_cropped, **self.display_data)
            self.reset_data()
            # np.save(self.buffer_fn, frame_cropped)
            time.sleep(interval)

    def signal_display(self, **signal):
        """Use PIPE to send data to the video server."""
        data_bytes = pickle.dumps(signal)
        self.video_player.stdin.flush()
        # first, send the length of the data_bytes for safe reading
        length_prefix = struct.pack('I', len(data_bytes))
        self.video_player.stdin.write(length_prefix)
        # then, send the data
        self.video_player.stdin.write(data_bytes)
        self.video_player.stdin.flush()

    # def update_display(self, img, headings, headings_smooth=None, com_shift=None):
    #     """Send images and heading data to the video server using stdin.PIPE."""
    #     data = {'img': img, 'headings': headings, 'headings_smooth': headings_smooth, 'com_shift': com_shift}
    #     self.signal_display(data)

    def reset_display(self):
        """Clear the plotted data and plots.

        By default, PIPEing trial data to the server will append it to a list 
        for each parameter. This function instructs the server to clear these
        lists.
        """
        self.reset_data()
        self.signal_display(reset=True)
        self.background = np.zeros((912, 1140, 4), dtype='uint8')

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
        self.reset_display()
        self.reset_data()
        if 'headings_fn' in dir(self):
            np.save(self.headings_fn, np.array([]))

    def save_frames(self, new_fn):
        """Save the captured frames as a new video."""
        if self.storing and not self.dummy:
            self.frames = np.concatenate(self.frames)
            self.frames = self.frames.reshape((self.frame_num, self.height, self.width))
            input_dict = {'-r': str(round(self.framerate)), '-hwaccel': 'cuda', '-hwaccel_output_format': 'cuda'}
            output_dict = {'-r': str(round(self.framerate)), '-c:v': 'h264_nvenc', '-preset':'slow', '-vf':'hue=s=0'}
            io.vwrite(new_fn, self.frames, inputdict=input_dict, outputdict=output_dict)
            print(f"{self.frame_num} frames saved in {new_fn}")

    def import_config(self, ring_thickness=3):
        # if 'last_imported' not in dir(self):
        #     self.last_import = 0
        # if os.path.getmtime(self.config_fn) > self.last_import:
        if True:
            # get the stored ring radii, threshold, and whether to invert
            info = configparser.ConfigParser()
            info.read(self.config_fn)
            if 'video_parameters' in info.keys():
                self.vid_params = info['video_parameters']
                for key, dtype in zip(['thresh', 'inner_r', 'outer_r', 'wing_r', 'invert', 'flipped', 'kalman_jerk_std', 'kalman_noise'],
                                        [int, float, float, float, bool, bool, float, float]):
                    if dtype == bool:
                        val = self.vid_params[key] == 'True'
                    else:
                        val = dtype(self.vid_params[key])
                    setattr(self, key, val)
                # get the inner ring indices and angles
                inner_inds = self.img_dists <= (self.inner_r + ring_thickness/2.)
                inner_inds *= self.img_dists > (self.inner_r - ring_thickness/2.)
                self.inner_inds = np.where(inner_inds)
                self.inner_angs = self.img_angs[self.inner_inds]
                self.inner_coords = self.img_coords[self.inner_inds]
                # get the outer ring indices and angles
                outer_inds = self.img_dists <= (self.outer_r + ring_thickness/2.)
                outer_inds *= self.img_dists > (self.outer_r - ring_thickness/2.)
                self.outer_inds = np.where(outer_inds)
                self.outer_angs = self.img_angs[self.outer_inds]
                self.outer_coords = self.img_coords[self.outer_inds]
                # get the wing ring indices and angles
                wing_inds = self.img_dists <= (self.wing_r + ring_thickness/2.)
                wing_inds *= self.img_dists > (self.wing_r - ring_thickness/2.)
                self.wing_inds = np.where(wing_inds)
                self.wing_angs = self.img_angs[self.wing_inds]
                self.wing_coords = self.img_coords[self.wing_inds]
                # update the kalman filter if values changed
                if 'kalman_filter' in dir(self):
                    std_changed = self.kalman_filter.jerk_std != self.kalman_jerk_std
                    noise_changed = self.kalman_filter.measurement_noise_x != self.kalman_noise
                    if std_changed or noise_changed:
                        self.kalman_setup()
            # import user-chosen experiment parameterstoo
            if "experiment_parameters" in info.keys():
                self.exp_params = info['experiment_parameters']

    def update_heading(self):
        # todo: speed up this stage using CUDA to process at a much higher framerate
        # todo: make a numpy array to store frames between outputs. this'll allow for faster 
        # indexing of the 3d array, grabbing the inner and outer rings more quickly
        # the center of mass calculations may be the slowest part here. CUDA should 
        # help with that.
        # print(self.buffer_stop - self.buffer_start)
        if self.playing and self.buffer_stop != self.buffer_start:
            # print(self.buffer_ind)
            # try:
            #     self.import_config()
            # except:
            #     pass
            """Use stored radii to approximate the heading."""
            # 0. grab current value of pertinent variabls
            thresh, invert = self.thresh, self.invert
            # use index numbers instead of a boolean array, because it's faster to translate
            # outer_inds, outer_angs = np.copy(self.outer_inds), np.copy(self.outer_angs)
            # inner_inds, inner_inds = np.copy(self.inner_inds), np.copy(self.inner_angs)
            # if self.outer_inds[0].size > self.outer_angs.size:
            #     outer_inds = (outer_inds[0][:outer_angs.size], outer_inds[1][:outer_angs.size])
            # outer_inds, outer_angs = self.outer_inds, self.outer_angs
            # inner_inds, inner_angs = self.inner_inds, self.inner_angs
            # frame = self.frame
            # frames = [self.frames_to_process[-1]]
            # frames = np.copy(self.buffer[0: self.buffer_ind])
            if self.buffer_stop < self.buffer_start:
                # frames = np.append(self.buffer[self.buffer_start:], self.buffer[:self.buffer_stop], axis=0)
                include = np.append(np.arange(self.buffer_start, len(self.buffer)), np.arange(0, self.buffer_stop))
            else:
                include = np.arange(self.buffer_start, self.buffer_stop)
                # frames = self.buffer[self.buffer_start:self.buffer_stop]
            frames = self.buffer[include]
            # print(frames.shape)
            self.buffer_start = self.buffer_stop
            # update the start and stop time points
            inner_ring = frames[:, self.inner_inds[0], self.inner_inds[1]]
                # print(self.inner_inds)
            # todo: grab important frame values because it will likely be replaced by the next frame soon
            # use the outer and inner inds
            # outer_ring = frames[:, outer_inds[0], outer_inds[1]]
            outer_ring = frames[:, self.outer_inds[0], self.outer_inds[1]]
                # print(self.outer_inds)
            # get boolean mask for center of mass calculation
            if invert:
                frame_mask = frames < thresh
            else:
                frame_mask = frames > thresh
            # diffs = []
            # headings = []
            # headings_smooth = []
            # self.frames_to_process = []
            # todo: use numpy operations instead of for loops through the frames

            # for frame in frames:
            # reshape frames in case they were added as flat arrays
            # if frames.ndim == 2:
            #     frames = frames.reshape(len(frames), self.height, self.width)
            # elif frames.ndim > 3:
            #     frames = frames[..., 0]
            # 1. get outer ring, which should include just the tail
            # and adjust inner and outer inds using the center of mass
            # if self.com_correction:
            #     diffs = []
            #     for mask in frame_mask:
            #         com = self.img_coords[mask].mean(0)
            #         diff = np.array([self.height/2, self.width/2]) - np.array(com)
            #         diffs += [diff]
            if self.flipped:
                # 2. if fly is forward leaning, then the head is in the outer ring
                # and we can ignore the inner ring
                if invert:
                    head = outer_ring < thresh
                else:
                    head = outer_ring > thresh
                if len(self.outer_angs) == head.shape[-1]:
                    outer_coords_filtered = self.outer_coords[np.newaxis] * head[..., np.newaxis]
                    mean_coord = outer_coords_filtered.sum(1) / head.sum(1)[:, np.newaxis]
                    headings = np.arctan2(mean_coord[..., 1], mean_coord[..., 0])
                    # headings += [heading]
            else:
                # 2. find the tail and head orientation by thresholding the outer ring
                # values and calculate the tail heading as the circular mean
                if invert:
                    tail = outer_ring < thresh
                else:
                    tail = outer_ring > thresh
                # print(f"outer_angs shape = {outer_angs.shape}, tail shape = {tail.shape}")
                # tail_angs = []
                # for ind in tail:
                #     tail_angs += [outer_angs[ind]]
                # tail_angs = [outer_angs[ind] for ind in tail]
                # tail_dir = scipy.stats.circmean(tail_angs, low=-np.pi, high=np.pi, axis=1)
                # tail_dir = np.zeros(len(tail), dtype=float)
                # for num, (ind) in enumerate(tail):
                #     if any(ind):
                #         # tail_dir[num] = scipy.stats.circmean(self.outer_angs[ind], low=-np.pi, high=np.pi)
                #         # consider that it may be faster to take the 2D mean of the x, y coordinates
                #         # and then find the angle of that, instead of using the circular mean
                #         mean_coord = self.outer_coords[ind].mean(0)
                #         tail_dir[num] = np.arctan2(mean_coord[1], mean_coord[0])
                outer_coords_filtered = self.outer_coords[np.newaxis] * tail[..., np.newaxis]
                mean_coord = outer_coords_filtered.sum(1) / tail.sum(1)[:, np.newaxis]
                tail_dir = np.arctan2(mean_coord[..., 1], mean_coord[...,0])
                # mean_coords = self.outer_coords[tail].mean((0, 1))
                # tail_dir = np.arctan2(mean_coords[1], mean_coords[0])
                # print(f"tail dir = {tail_dir}")
                # todo: fix the rest of this algorithm 
                # the head direction is the polar opposite of the tail
                head_dir = tail_dir + np.pi
                # wrap around if the head direction is outside of [-pi, pi]
                out_of_bounds = head_dir > np.pi
                head_dir[out_of_bounds] -= 2 * np.pi    
                # 3. get bounds of head angles, ignoring angles within +/- 90 degrees of the tail
                # lower_bounds, upper_bounds = [head_dir - np.pi / 2], [head_dir + np.pi / 2]
                # wrap bounds if they go outside of [-pi, pi]
                # todo: use array operations instead of for loops to get the pi range of 
                # subtracting head_dir from self.inner_angs should move the head to 0, so we can use lower and upper bounds of -pi/2 and pi/2
                inner_angs_centered = self.inner_angs[np.newaxis] - head_dir[:, np.newaxis]
                # wrap around any values outside of [-pi, pi]
                inner_angs_centered[inner_angs_centered < -np.pi] += 2 * np.pi
                inner_angs_centered[inner_angs_centered > np.pi] -= 2 * np.pi
                # include only those angles that are within +/- pi/2 of the head
                include = np.abs(inner_angs_centered) < np.pi/2
                # if lower_bounds[0] < -np.pi:
                #     lb = np.copy(lower_bounds[0])
                #     lower_bounds[0] = -np.pi
                #     lower_bounds += [lb + 2 * np.pi]
                #     upper_bounds += [np.pi]
                # elif upper_bounds[0] > np.pi:
                #     ub = np.copy(upper_bounds[0])
                #     upper_bounds[0] = np.pi
                #     upper_bounds += [ub - 2 * np.pi]
                #     lower_bounds += [-np.pi]
                # lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)
                # 4. calculate the heading within the lower and upper bounds
                # inner_inds, inner_angs = np.where(self.inner_inds), self.inner_angs
                # if inner_inds[0].size > inner_angs.size:
                #     inner_inds = (inner_inds[0][:inner_angs.size], inner_inds[1][:inner_angs.size])
                # while np.any(inner_inds[0] < -100):
                #     inner_inds, inner_angs = np.where(self.inner_inds), self.inner_angs
                # include = np.zeros(len(inner_angs), dtype=bool)
                # for lower, upper in zip(lower_bounds, upper_bounds):
                #     include += (inner_angs > lower) * (inner_angs < upper)
                headings = np.zeros(len(tail), dtype=float)
                headings.fill(np.nan)
                if np.any(include):
                    if self.invert:
                        head = inner_ring < thresh
                    else:
                        head = inner_ring > thresh
                    head[include == 0] = False
                    # use the coordinates to speed up the angle calculation
                    inner_coords_filtered = self.inner_coords[np.newaxis] * head[..., np.newaxis]
                    mean_coord = inner_coords_filtered.sum(1) / head.sum(1)[:, np.newaxis]
                    headings = np.arctan2(mean_coord[..., 1], mean_coord[..., 0])
                    # for num, (ind) in enumerate(head):
                    #     if any(ind):
                    #         # angs = self.inner_angs[ind]
                    #         # heading = scipy.stats.circmean(angs, low=-np.pi, high=np.pi)
                    #         # headings[num] = scipy.stats.circmean(self.inner_angs[ind], low=-np.pi, high=np.pi)
                    #         head_mean = self.inner_coords[ind].mean(0)
                    #         headings[num] = np.arctan2(head_mean[1], head_mean[0])
            # headings = np.array(headings)
            if self.com_correction:
                diffs = []
                for mask in frame_mask:
                    diff = -self.img_coords[mask].mean(0)[::-1]
                    # center = np.array([self.height/2, self.width/2])
                    # diff = np.array([self.height/2, self.width/2]) - np.array(com)
                    diffs += [diff]
                diffs = np.array(diffs)
                # convert from heading angle to head position for each heading
                heading_pos = self.inner_r * np.array([np.sin(headings), np.cos(headings)]).T
                # calculate direction vector between the center of the fly and the head
                direction = heading_pos + diffs
                headings = np.arctan2(direction[:, 0], direction[:, 1])
                if not np.any(np.isnan(direction[-1])):
                    # store the shift in the center of mass for plotting
                    self.com_shift = np.round(diffs[-1]).astype(int)
                else:
                    self.com_shift = np.zeros(2)
            # center and wrap the heading
            headings -= np.pi/2
            out_of_bounds = headings < -np.pi
            headings[out_of_bounds] += 2 * np.pi
            # if headings < -np.pi:
            #     headings += 2 * np.pi
            # store
            self.heading = copy.copy(headings[-1])
            self.headings = np.append(self.headings, headings[-1:])
            # headings += [heading]
            if self.kalman:
                headings_smooth = []
                heading_smooth = self.heading
                # use kalman filter, only if heading is not np.nan
                if not np.isnan(heading_smooth):
                    # apply the kalman filter to the headings and output both the 
                    for heading in headings: 
                        self.kalman_filter.store(heading)
                        headings_smooth += [self.kalman_filter.predict()]
                        # try:
                        #     self.kalman_filter.store(heading)
                        # except:
                        #     self.kalman_setup()
                        #     self.kalman_filter.store(heading)
                    heading_smooth = headings_smooth[-1]
                    diff = heading - heading_smooth
                    if abs(diff) > 2*np.pi:
                        new_diff = diff // 2*np.pi
                        heading_smooth += new_diff
                    # check if acceleration is above a threshold
                    if len(self.kalman_filter.record) > 10:
                        speed = abs(np.diff(self.kalman_filter.record[-4:-1], axis=0))
                        acceleration = abs(np.diff(speed))
                        acceleration *= self.framerate**2
                        if np.any(np.isnan(speed)):
                            self.kalman_setup()
                            self.is_saccading = False
                        else:                        
                            speed = np.nanmean(speed)
                            # if sacccading, then check if the acceleration is below a threshold and switch
                            if self.is_saccading:
                                if (speed < .1*self.speed_thresh) and (self.frame_num - self.saccade_frame > 10):
                                    self.is_saccading = False
                                    self.kalman_setup()
                                elif speed > self.speed_thresh:
                                    self.speed_thresh = speed
                            # when not saccading, check if the acceleration is above a threshold and switch
                            else:
                                # heading_smooth = self.kalman_filter.predict()
                                self.heading = heading_smooth
                                if acceleration > self.acceleration_thresh:
                                    self.speed_thresh = speed
                                    if 'saccade_num' not in dir(self):
                                        self.saccade_num = 0
                                    self.saccade_num += 1
                                    self.is_saccading = True
                                    self.saccade_frame = np.copy(self.frame_num)
                                    # print(f'saccade # {self.saccade_num}, frame {self.frame_num}')
                else:
                    for heading in headings[-1:]:
                        headings_smooth += [heading]
                self.headings_smooth = np.append(self.headings_smooth, headings_smooth[-1:])
                # headings_smooth += [heading_smooth]
            if self.wing_analysis:
                if self.frame_num < 5:
                    heading_smooth = np.array
                else:
                    headings_smooth = np.array(headings_smooth)
                # grab the ring of values from the frames pertaining to the wingbeats
                wing_vals = frames[:, self.wing_inds[0], self.wing_inds[1]]
                wing_angs = self.wing_angs
                wing_angs_centered = wing_angs[np.newaxis] - headings[:, np.newaxis] + np.pi/2
                wing_angs_centered[wing_angs_centered < -np.pi] += 2 * np.pi
                wing_angs_centered[wing_angs_centered >= np.pi] -= 2 * np.pi
                # test: show the image with superimposed rings 
                frame = frames[0]
                angs = wing_angs_centered[0]
                vals = wing_vals[0]
                # plt.imshow(frame, cmap='gray')
                # plt.scatter(self.wing_inds[1], self.wing_inds[0], c=vals, marker='.', alpha=.25)
                # plt.show()
                # test: plot the wing angles and corresponding values
                # for angs, vals, color in zip(wing_angs_centered, wing_vals, plt.cm.viridis(np.linspace(0, 1, 8))): order = np.argsort(angs); plt.plot(angs[order], vals[order], color=color)
                # plt.show()
                # I have 2 ideas for how to get the wing edge: 
                # 1) use the raw values, find the bottom 3 local minima, and use the one with the widest spread
                # 2) remove the 'background' (by averaging across all frames) first, and then find the local minima ...
                # 1 would be faster, but may make mistakes due to fluctuations in the background, which #2 would avoid
                # todo: #1 failed, so try #2
                # 1) remove the background
                background = np.nanmean(wing_vals, axis=0)
                wing_vals_diff = wing_vals - background
                wing_vals_diff[wing_vals_diff > 0] = 0
                # 2) split the values into the left and right halves. the right wing corresponds to positive, centered angles
                # left_half = wing_angs_centered >= np.pi
                # right_half = wing_angs_centered < np.pi
                left_half = wing_angs_centered >= 0
                right_half = wing_angs_centered < 0
                # 3) find the local minima for each half
                left_peaks, right_peaks = [], []
                for half, edge_variable, storage in zip([left_half, right_half], ['right_bases', 'left_bases'], [left_peaks, right_peaks]):
                    for vals, angs, include, frame, heading in zip(-wing_vals_diff, wing_angs_centered, half, frames, headings + np.pi):
                        order = np.argsort(angs[include])
                        # find the local minima
                        # peaks = signal.find_peaks_cwt(vals[half][order], widths=100)
                        # remove small differences before applying the peak finder
                        clean_vals = vals[include][order]
                        # clean_vals[clean_vals < 2] = 0
                        ## to use the wing envelope:
                        # peaks, fitness = scipy.signal.find_peaks(clean_vals, distance=500, prominence=2)
                        # # edges = fitness[edge_variable]
                        # edges = peaks
                        # edge = np.nan
                        # if len(edges) > 1:
                        #     # find the peak with the greatest prominence
                        #     edge = edges[np.argmax(fitness['prominences'])]
                        # elif len(edges) == 1:
                        #     edge = edges[0]
                        # test: plot the wing angles and corresponding values
                        # plt.plot(angs[include][order], vals[include][order])
                        # plt.plot(angs[include][order], clean_vals)
                        # if len(edges) > 0:
                        #     plt.axvline(angs[include][order][edge])
                        # plt.figure()
                        # plt.imshow(frame, cmap='gray', origin='lower')
                        # plt.title(edge_variable)
                        # plt.scatter(self.wing_inds[1][include], self.wing_inds[0][include], c=vals[include], marker='.', alpha=.25)
                        # if not np.isnan(edge):
                        #     plt.scatter(self.wing_inds[1][include][order][edge], self.wing_inds[0][include][order][edge], c='r', marker='o')
                        # plot a vector in the direction of the heading
                        # r = 100
                        # cx, cy = self.width/2, self.height/2
                        # rx, ry = r * np.cos(heading + np.pi/4), r * np.sin(heading + np.pi/4)
                        # rx, ry = r * np.cos(heading + 3*np.pi/2), r * np.sin(heading + 3*np.pi/2)
                        # plt.plot([cx, rx+cx], [cy, ry+cy])
                        # if more than 1 peak, then take the one with the greatest prominence
                        inds = np.arange(len(clean_vals))
                        edge = int(np.round(np.sum(inds * clean_vals)/clean_vals.sum()))
                        # # store the peaks
                        if np.isnan(edge):
                            edge_ang = np.nan
                        else:
                            edge_ang = angs[include][order][edge]
                        # edge = int(round(np.mean(np.arange(len(clean_vals)) * clean_vals)))
                        # edge_ang = (angs[include][order] * clean_vals).sum()/(clean_vals.sum())
                        storage += [edge_ang]
                #         plt.savefig("test.png")
                # breakpoint()
            if 'display_data' in dir(self):
                if self.wing_analysis:
                    left_peaks = -np.array(left_peaks)
                    right_peaks = -np.array(right_peaks)
                    self.display_data['wing_left'] += [right_peaks]
                    self.display_data['wing_right'] += [left_peaks]
                    # self.display_data['wing_left'] += [[np.pi/2]]
                    # self.display_data['wing_right'] += [[np.pi/2]]
                    # adding pi/2 will make 0 the direction of the head
                    # -pi/2 is the right side and pi/2 is the left side
                self.display_data['heading_smooth'] += [headings_smooth]
                self.display_data['com_shift'] += [diffs]
                self.display_data['heading'] += [headings]
            # now add the frames to the queue for recording data
            # for frame in np.copy(frames):
            for frame in frames:
                if self.storing:
                    self.frames.put(frame)
                    self.frame_num += 1
            if self.kalman:
                return self.headings_smooth[-1]
            else:
                return self.headings[-1]
        else:
            if self.kalman:
                heading = self.kalman_filter.predict()
            elif len(self.headings) > 0:
                heading = self.headings[-1]
            else:
                heading = np.nan
            self.headings = np.append(self.headings, heading)
            return heading

    def reset_data(self):
        if 'display_data' in dir(self):
            for key in self.display_data.keys():
                if key != 'img':
                    self.display_data[key] = []

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
        self.reset_data()
        self.reset_display()


class TrackingTrial():
    def __init__(self, camera, window, dirname, kalman=False, gui_config_fn="video_server.config"):
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
        gui_config_fn : str, default="video_server.config"
            Filename of the config file for the GUI. This is used to
            save experimental parameters selected by the user.
        """
        self.config_fn = os.path.abspath(gui_config_fn)
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
        # make an empty list for test data
        self.tests = []
        self.yaws = []  # the list of yaw arrays per test from holocube
        self.headings = []  # the list of headings per test from the camera
        self.headings_smooth = []  # the list of headings per test from the camera
        self.heading = 0
        self.virtual_headings_test = []  # the list of headings per test from the camera
        self.virtual_headings = []  # the list of headings per test from the camera
        self.realtime = []   # list of timestamps for each stored frame
        self.test_info = {}  # the experiment parameters per test

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
            # store the experiment parameters saved in video_server.config
            if "exp_params" in dir(self.camera):
                for key, val in self.camera.exp_params.items():
                    self.h5_file.attrs[key] = val


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
            self.yaws += [arr[0]]
            if len(arr) > 1:
                self.realtime += [arr[1]]
            self.headings += [self.camera.get_headings()]
            self.headings_smooth += [self.camera.get_headings_smooth()]
            # try:
            #     self.virtual_headings += [self.virtual_objects['fly_heading'].get_angles()]
            # except:
            #     breakpoint()
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
        if self.camera.capturing and not self.camera.dummy:
            if callable(value):
                value = value()
            self.h5_file.attrs[key] = value

    def save(self):
        """Store the test data and information into the H5 dataset."""
        # if self.camera.capturing and not self.camera.dummy:
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
                    [self.yaws, self.realtime, self.headings, self.headings_smooth, self.virtual_headings],
                    ['yaw', 'realtime', 'camera_heading', 'camera_heading_smooth' 'virtual_angle']):
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
                    try:
                        sub_val = sub_val[0]
                    except:
                        breakpoint()
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
                    print(param)
                    self.h5_file.create_dataset(param, data=np.squeeze(arr))
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
        is_nan = heading == np.nan
        is_none = heading is None
        if is_nan or is_none:
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

