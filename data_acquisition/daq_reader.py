"""Setup communication with a National Instruments DAQ.

todo:
-make buttons for important functions:
    -save current buffer
    -change xmin using key presses like in Jamie's script
    -incorporate with holocube
"""
from datetime import datetime
from functools import partial
import threading
import h5py
import numpy as np
import nidaqmx
import sys
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from nidaqmx import stream_readers, stream_writers, system, constants
import configparser
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import os
import time

class DAQReader():
    def __init__(self, config_fn="./DAQ.ini", storage_directory="./"):
        """Interface with a National Instruments Data AcQuisition Card.


        Parameters
        ----------
        :param config_fn: str default="./DAQ.ini"
            Filename of the configuration file detailing the desired channels
            for recording.
        """
        # read config file
        config = configparser.ConfigParser()
        config.read(config_fn)
        # save the directory for storing data
        self.storage_directory = storage_directory
        if not os.path.isdir(self.storage_directory):
            os.mkdir(self.storage_directory)
        # get device information and setup the connection
        self.device_info = config['device']
        self.device_name = self.device_info['name']
        self.buffer_size = int(self.device_info['buffer_size'])
        self.sample_rate = int(self.device_info['sample_rate'])
        self.display_rate = int(self.device_info['display_rate'])
        self.projector_rate = int(self.device_info['projector_rate'])
        # setup the device settings
        self.device = system.Device(self.device_name)
        # setup the analog channels
        # get the list of digital and analog channels to record from
        self.channels_digital = dict(config['digital_channels'])
        self.channels_analog = dict(config['analog_channels'])
        self.buffer_analog, self.buffer_digital = [], []
        # set the sample rate and bin size for rendering at the desired display rate
        self.bin_size = round(float(self.sample_rate)/float(self.display_rate))
        # self.bin_size = 1
        for input_type, channels in zip(['analog', 'digital'], [self.channels_analog, self.channels_digital]):
            if len(channels) > 0:
                task = nidaqmx.Task()
                self.__setattr__(f"task_{input_type}", task)
                for name, val in channels.items():
                    # add analog input channel
                    if input_type == 'analog':
                        task.ai_channels.add_ai_voltage_chan(val, min_val=-5, max_val=5)
                        reader = stream_readers.AnalogMultiChannelReader
                    else:
                        task.di_channels.add_di_chan(val, line_grouping=constants.LineGrouping.CHAN_FOR_ALL_LINES)
                        reader = stream_readers.DigitalMultiChannelReader
                # make a large buffer for storing the streamed data
                self.__setattr__(f'buffer_{input_type}', np.zeros(
                    (len(channels), self.buffer_size)))
                # setup the input frame rate
                if input_type == 'analog':
                    task.timing.cfg_samp_clk_timing(
                        rate=self.sample_rate, sample_mode=constants.AcquisitionType.CONTINUOUS,
                        samps_per_chan=self.bin_size)
                # else:
                #     task.timing.samp_timing_type = constants.SampleTimingType.ON_DEMAND
                #     task.timing.cfg_samp_clk_timing(
                #         self.sample_rate,
                #         samps_per_chan=self.bin_size)
                # initialize the stream reader
                self.__setattr__(f'input_{input_type}', reader(task.in_stream))
                # make temporary buffer
                if input_type == 'analog':
                    self.__setattr__(f'temp_buffer_{input_type}', np.zeros((len(channels), self.bin_size)))
                else:
                    self.__setattr__(f'temp_buffer_{input_type}', np.zeros((len(channels), self.bin_size), dtype='uint8'))
        # plot the channels
        self.scale = max(round(float(self.buffer_size) / 10000.), 1)
        print(f"bin size = {self.bin_size}")
        # start a circular indexing
        self.current_frame_analog = 0
        self.current_frame_digital = 0
        # plot the figure
        self.plot()
        # setup autoscaling whenever the user changes the x-range
        self.plots[0].sigXRangeChanged.connect(self.set_scale)
        # low_time is the duration of off and high_time is the duration of on signal
        # start a PWM signal using ctr0, which outputs to PFI12 based on NiMAX
        # add callback to the task
        self.is_reading = False
        self.start_readers()
        # make a QTimer to update the plot
        self.plotter_thread()
        # self.is_recording = False
        self.start_camera()

    def start_camera(self, frame_ratio=1, duty_cycle=.1):
        """Start synchronized camera acquisition.

        Parameters
        ----------
        frame_ratio : int, default=1
            The number of camera frames per each projector frame.
        duty_cycle : float, default=.5
            The proportion of each frame interval that the camera will sense light.

        """
        # Create a new task for the digital output channel
        self.camera_task_output = nidaqmx.Task()
        # Add a digital output channel to the task
        self.camera_task_output.ao_channels.add_ao_voltage_chan('MagnoDAQ/ao0', min_val=0, max_val=3.3)
        # change the sample rate
        self.output_rate = self.sample_rate
        self.output_bin_size = int(round(self.bin_size * self.output_rate/self.sample_rate))
        self.camera_task_output.out_stream.output_buf_size = self.output_bin_size
        self.camera_task_output.timing.cfg_samp_clk_timing(rate=self.output_rate, samps_per_chan=self.output_bin_size)
        # make a stream writer
        self.camera_writer = stream_writers.AnalogSingleChannelWriter(self.camera_task_output.out_stream, auto_start=True)
        # start camera 
        self.set_camera_exposure(frame_ratio, duty_cycle)
        # self.camera_thread = threading.Thread(target=self.camera_loop, name='camera_loop')
        self.is_recording = True
        # self.camera_thread.start()
        while self.is_recording:
            if np.any(self.temp_buffer_analog[0] > 1):
                try:
                    self.camera_writer.write_many_sample(self.camera_pulse)
                except:
                    breakpoint()
                # self.camera_task_output.write([3.3, 3.3])
                # self.camera_task_output.wait_until_done()
                # self.camera_task_output.write(self.camera_pulse, auto_start=True)
                # time.sleep(self.exposure)
                # self.camera_task_output.write([0, 0])
                time.sleep(.1)

    def camera_loop(self):
        while self.is_recording:
            if np.any(self.temp_buffer_analog[0] > 1):
                # self.camera_writer.write_many_sample(self.camera_pulse[:-1])
                # self.camera_task_output.write(3.3)
                self.camera_task_output.write(self.camera_pulse, auto_start=True)
                # time.sleep(self.exposure)
                # self.camera_task_output.write(0)
                time.sleep(.1)

    def set_camera_exposure(self, frame_ratio=1, duty_cycle=.1):
        self.frame_ratio = frame_ratio
        self.camera_fps =  self.frame_ratio * self.projector_rate
        interval = 1./self.camera_fps
        self.exposure = duty_cycle * interval
        print(f"exposure: {self.exposure}, camera fps: {self.camera_fps}")
        # todo: make a pulse bin
        self.camera_pulse = np.zeros(self.output_bin_size)
        dur = int(round(self.exposure * self.output_rate))
        self.camera_pulse[:dur] = 3.3

    def start_readers(self):
        """Make a separate thread to repeatedly run self.read()."""
        # set is_reading to true 
        self.is_reading = True
        # make the thread for reading analog data 
        if len(self.channels_analog) > 0:
            self.analog_reader = threading.Thread(target=self.reader_analog, name='reader_analog')
            self.analog_reader.start()
        # make the thread for reading digital data 
        if len(self.channels_digital) > 0:
            self.digital_reader = threading.Thread(target=self.reader_digital, name='reader_digital')
            self.digital_reader.start()

    def plotter_thread(self):
        """Make a QTimer to repeatedly update the plot."""
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.setInterval(round(1000./(self.display_rate)))
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start()

    def reader_digital(self):
        """Start process for storing the temporary buffer of digital data."""
        while self.is_reading:
            # grab the temporary bin
            self.grab_bin_digital()
            # check if any triggers were set off
            if 'camera_task' in dir(self):
                if np.any(self.temp_buffer_digital[0] > 1):
                    self.expose_camera()
            # calculate the maximum frame included
            end_frame = self.current_frame_digital + self.bin_size
            if end_frame > self.buffer_size:
                diff = end_frame - self.buffer_size
                # input digital values
                self.buffer_digital[:, self.current_frame_digital:] = self.temp_buffer_digital[:, :-diff]
                self.buffer_digital[:, :diff] = self.temp_buffer_digital[:, -diff:]
            else:
                # input digital values
                self.buffer_digital[:, self.current_frame_digital:end_frame] = self.temp_buffer_digital
            # update current frame number
            self.current_frame_digital += self.bin_size
            self.current_frame_digital %= self.buffer_size

    def reader_analog(self):
        """Start process for storing the temporary buffer of analog data."""
        while self.is_reading:
            # grab the temporary bin
            self.grab_bin_analog()
            # calculate the maximum frame included
            end_frame = self.current_frame_analog + self.bin_size
            if end_frame > self.buffer_size:
                diff = end_frame - self.buffer_size
                if 'temp_buffer_analog' in dir(self):
                    # input analog values
                    self.buffer_analog[:, self.current_frame_analog:] = self.temp_buffer_analog[:, :-diff]
                    self.buffer_analog[:, :diff] = self.temp_buffer_analog[:, -diff:]
            else:
                # input analog and digital values
                if 'temp_buffer_analog' in dir(self):
                    self.buffer_analog[:, self.current_frame_analog:end_frame] = self.temp_buffer_analog
            # update current frame number
            self.current_frame_analog += self.bin_size
            self.current_frame_analog %= self.buffer_size

    def grab_bin_analog(self):
        """Grab a bin of analog samples to the desired sample rate."""
        # read from analog channels
        self.input_analog.read_many_sample(
            data=self.temp_buffer_analog,
            number_of_samples_per_channel=self.bin_size)

    def grab_bin_digital(self):
        """Grab a bin of frames to the desired sample rate."""
        # read from digital channels
        self.input_digital.read_many_sample_port_byte(
            data=self.temp_buffer_digital,
            number_of_samples_per_channel=self.bin_size)

    def plot(self):
        """Plot the incoming stream from the DAQ."""
        # get the time value for each frame
        self.times = np.linspace(
            -self.buffer_size / self.sample_rate, 0, self.buffer_size)
        # initialize the application, window, and plot
        self.app = QtWidgets.QApplication(sys.argv)
        self.view = pg.GraphicsView()
        self.layout = pg.GraphicsLayout()
        self.view.setCentralItem(self.layout)
        self.view.show()
        self.view.resize(600, len(self.channels_analog) * 200)
        # add plots for each channel
        self.plots = []
        first_plot = None  # use first plot to link all the x-axes
        num = 0
        for channels, buffer in zip(
            [self.channels_analog, self.channels_digital], 
            [self.buffer_analog, self.buffer_digital]):
            for channel, vals in zip(channels.keys(), buffer):
                plot = self.layout.addPlot(num, 0, x=self.times, y=vals,
                                        name=channel,
                                        title=channel.replace("_", " "))
                # make lineplot
                plot.dataItems[0].setData(x=self.times[::self.scale], y=vals[::self.scale])
                plot.showGrid(x=True, y=True, alpha=.5)
                self.plots += [plot]
                if first_plot is None:
                    first_plot = plot
                else:
                    plot.setXLink(first_plot)
                num += 1
        # make a vertical layout for the buttons
        # self.layout_buttons = QtWidgets.QVBoxLayout()
        # make a proxy widget to place the graphic item
        self.button_capture_proxy = QtWidgets.QGraphicsProxyWidget()
        self.button_capture = QtWidgets.QPushButton('capture')
        self.button_capture.clicked.connect(self.capture)
        self.button_capture_proxy.setWidget(self.button_capture)
        # add the button to the window layout
        self.layout.addItem(self.button_capture_proxy)
        # make a proxy widget to place the graphic item
        self.button_preview_proxy = QtWidgets.QGraphicsProxyWidget()
        self.button_preview = QtWidgets.QPushButton('capture +\npreview')
        self.button_preview.clicked.connect(self.preview)
        self.button_preview_proxy.setWidget(self.button_preview)
        # add the button to the window layout
        self.layout.addItem(self.button_preview_proxy)

    def update_plot(self):
        """Update the plots by moving the current frame index."""
        num = 0
        for buffer, current_frame in zip(
            [self.buffer_analog, self.buffer_digital], 
            [self.current_frame_analog, self.current_frame_digital]):
            for vals in buffer:
                plot = self.plots[num]
                ordered_vals = np.append(
                    vals[current_frame:],
                    vals[:current_frame], axis=0)
                plot.dataItems[0].setData(x=self.times[::self.scale],
                                          y=ordered_vals[::self.scale])
                num += 1

    def set_scale(self, *args, num_points=100000):
        """Determine the resolution scale for plotting the data.

        Since we cannot see differences beyond a certain resolution,
        we can save processing time by plotting at a lower resolution,
        based on the xrange.
        """
        # get the x-range from the plot and use it to scale the plot resolution
        (xmin, xmax), (ymin, ymax) = self.plots[0].viewRange()
        included = self.times[(self.times > xmin) * (self.times < xmax)]
        num_frames = len(included)
        self.scale = max(round(num_frames / num_points), 1)
        print(f"plot scale = {self.scale}")

    def capture(self):
        now = datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        new_fn = os.path.join(self.storage_directory, timestamp + ".hdf5")
        # store in an hdf5 file with the timestamp as the filename
        h5_file = h5py.File(new_fn, 'w')
        # store the buffer
        for channels, buffer in zip([self.channels_analog, self.channels_digital], [self.buffer_analog, self.buffer_digital]):
            for (name, port), vals in zip(
                    channels.items(), buffer):
                h5_file[name] = vals
        # and the time marks
        h5_file['time'] = self.times
        # store channel info
        h5_file['analog_info'] = tuple(self.channels_analog)
        h5_file['digital_info'] = tuple(self.channels_digital)
        # notify the user
        print(f"buffer captured in {new_fn}")

    def preview(self):
        """Capture the current buffer and preview it in a matplotlib graph."""
        self.capture()
        matplotlib.use('Qt5Agg')
        # grab the values from the buffer
        buffer_analog = np.copy(self.buffer_analog)
        buffer_digital = np.copy(self.buffer_digital)
        # make a matplotlib figure
        num_rows = len(buffer_analog) + len(buffer_digital)
        fig, axes = plt.subplots(nrows=num_rows, sharex=True)
        plt.subplots_adjust(bottom=.2)
        if num_rows == 1:
            axes = [axes]
        # for each channel, plot the buffer
        num = 0
        for num, (buffer, channels) in enumerate(zip([buffer_analog, buffer_digital], [self.channels_analog, self.channels_digital])):
            for vals, channel in zip(buffer, channels):
                ax = axes[num]
                ax.plot(self.times, vals)
                ax.set_ylabel(channel.replace("_", ""))
                num += 1
        # format and plot it
        plt.tight_layout()
        plt.show()

daq = DAQReader(storage_directory='test_data')
# start plotting the traces
daq.app.exec()

# start a QTimer process to run daq.read until stopped.
# start a thread to run daq.update_plot when it receives a signal from daq.read

threshed = daq.buffer_analog[0] > .5
edges = np.diff(threshed.astype(int))
matplotlib.use('Qt5Agg')
plt.plot(edges)
# plt.xlim(0, 1000)
plt.show()
edges = np.where(edges == 1)[0]
# camera rate = 100 frame per second
# based on our measurements, 1 camera frame = 29 DAQ samples
# what is the duration of one DAQ sample?
# => .001 seconds per 29 DAQ samples
# => DAQ rate = 29/.001
# frame_lengths = np.diff(edges)
# frame_duration_camera = .01/5       # seconds
# num_frames_DAQ = np.median(frame_lengths)
# DAQ_framerate = num_frames_DAQ / frame_duration_camera
# print(f"DAQ framerate = {DAQ_framerate}")
# the framerate on the DAQ is roughly 50,000 Hz
# let's test for the framerate of the camera
frame_lengths = np.diff(edges)   # in DAQ samples per camera frame
frame_rate = 1. / frame_lengths  # in camera frames per DAQ sample
sample_rate = daq.sample_rate              # DAQ samples per second
frame_rate = frame_rate * sample_rate
print(f"Camera framerate = {np.median(frame_rate)} fps")
