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
from nidaqmx import stream_readers
from nidaqmx import system
import configparser
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import os

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
        # setup the device settings
        self.device = system.Device(self.device_name)
        # setup the analog and digital channels
        # get the list of digital and analog channels to record from
        self.channels_digital = dict(config['digital_channels'])
        self.channels_analog = dict(config['analog_channels'])
        # todo: add feature for digital channels. analog has much better resolution
        # add channels to different digital and analog tasks
        # self.task_digital = nidaqmx.Task()
        # add channels to the task using the configuration data
        # for name, val in self.channels_digital.items():
        #     # add digital input channel
        #     self.task_digital.di_channels.add_di_chan(
        #         val, line_grouping=LineGrouping.CHAN_PER_LINE)
        self.task_analog = nidaqmx.Task()
        for name, val in self.channels_analog.items():
            # add analog input channel
            self.task_analog.ai_channels.add_ai_voltage_chan(val)
        # todo: set the data transfer method to USB bulk
        # nidaqmx.constants.DataTransferActiveTransferMode(12590)
        # try https://nspyre.readthedocs.io/en/latest/guides/ni-daqmx.html
        # make separate digital and analog buffers
        # if len(self.channels_digital) > 0:
        #     self.buffer_digital = np.zeros(
        #         (len(self.channels_digital), self.buffer_size),
        #         dtype='uint8')
        if len(self.channels_analog) > 0:
            self.buffer_analog = np.zeros(
                (len(self.channels_analog), self.buffer_size))
        # initialize the analog and digital stream readers
        self.reader_analog = stream_readers.AnalogMultiChannelReader(
             self.task_analog.in_stream)
        # self.reader_digital = stream_readers.DigitalMultiChannelReader(
        #     self.task_digital.in_stream)
        # start a circular index
        self.current_frame = 0
        self.plotted_frame = 0
        # the DAQ has a fixed sample rate of ~50000 Hz
        # figure out the necessary bin size to get our desired
        # sample rate
        # self.bin_size = round(float(self.sample_rate)/float(self.display_rate))
        self.bin_size = 200
        #self.bin_size = int(2048 / len(self.channels_analog))
        # make a temporary buffer
        self.temp_buffer_analog = np.zeros(
            (len(self.channels_analog), self.bin_size))
        # plot the channels
        self.scale = round(float(self.buffer_size) / 10000.)
        self.plot()
        # setup autoscaling whenever the user changes the x-range
        self.plots[0].sigXRangeChanged.connect(self.set_scale)
        # start grabbing data from the channels
        self.reader_thread()
        # self.reader()
        # make a QTimer to update the plot
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.setInterval(round(1000./(2 * self.display_rate)))
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start()

    def reader_thread(self):
        """Make a separate thread to repeatedly run self.read()."""
        # make the thread
        thread = threading.Thread(target=self.reader, name='reader')
        thread.start()

    def reader(self):
        """Store binned frames from the temporary buffer
        """
        while True:
            # grab the temporary bin
            self.grab_bin()
            # calculate the maximum frame included
            end_frame = self.current_frame + self.bin_size
            if end_frame > self.buffer_size:
                diff = end_frame - self.buffer_size
                self.buffer_analog[:, self.current_frame:] = self.temp_buffer_analog[:, :-diff]
                self.buffer_analog[:, :diff] = self.temp_buffer_analog[:, -diff:]
            else:
                self.buffer_analog[
                :, self.current_frame:end_frame] = self.temp_buffer_analog
            # update current frame number
            self.current_frame += self.bin_size
            self.current_frame %= self.buffer_size
            # self.update_plot()

    def grab_bin(self):
        """Grab a bin of frames to the desired sample rate."""
        # todo: add digital channels
        if len(self.channels_analog) > 0:
            self.reader_analog.read_many_sample(
                data=self.temp_buffer_analog,
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
        for num, (channel, vals) in enumerate(
                zip(self.channels_analog.keys(), self.buffer_analog)):
            plot = self.layout.addPlot(num, 0, x=self.times, y=vals,
                                       name=channel,
                                       title=channel.replace("_", " "))
            # re-plot
            plot.dataItems[0].setData(x=self.times[::self.scale], y=vals[::self.scale])
            plot.showGrid(x=True, y=True, alpha=.5)
            self.plots += [plot]
            if first_plot is None:
                first_plot = plot
            else:
                plot.setXLink(first_plot)
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
        """Update the plots by adding moving the current frame index."""
        if self.plotted_frame != self.current_frame:
            for num, (plot, vals) in enumerate(zip(self.plots, self.buffer_analog)):
                ordered_vals = np.append(
                    vals[self.current_frame:],
                    vals[:self.current_frame])
                plot.dataItems[0].setData(x=self.times[::self.scale],
                                          y=ordered_vals[::self.scale])
            self.plotted_frame = self.current_frame

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

    def capture(self, *args, buffer=None, timestamp=None):
        """Capture the current buffer and store to a h5 file."""
        # grab the values from the buffer
        buffer = np.copy(self.buffer_analog)
        # make a filename based on the current time
        now = datetime.now()
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        self.save_data(buffer=buffer, timestamp=timestamp)

    def save_data(self, buffer, timestamp):
        new_fn = os.path.join(self.storage_directory, timestamp + ".hdf5")
        # store in an hdf5 file with the timestamp as the filename
        h5_file = h5py.File(new_fn, 'w')
        # store the buffer
        for (name, port), vals in zip(
                self.channels_analog.items(), buffer):
            h5_file[name] = vals
        # and the time marks
        h5_file['time'] = self.times
        # store channel info
        h5_file['channel_info'] = tuple(self.channels_analog)
        # notify the user
        print(f"buffer captured in {new_fn}")

    def preview(self):
        """Capture the current buffer and preview it in a matplotlib graph."""
        self.capture()
        matplotlib.use('Qt5Agg')
        # grab the values from the buffer
        buffer = np.copy(self.buffer_analog)
        # make a matplotlib figure
        fig, axes = plt.subplots(nrows=len(self.channels_analog), sharex=True)
        plt.subplots_adjust(bottom=.2)
        if len(self.channels_analog) == 1:
            axes = [axes]
        # for each channel, plot the buffer
        for ax, vals, channel in zip(axes, buffer, self.channels_analog.keys()):
            ax.plot(self.times, vals)
            ax.set_ylabel(channel.replace("_", ""))
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
sample_rate = 50000              # DAQ samples per second
frame_rate = frame_rate * sample_rate
print(f"Camera framerate = {np.median(frame_rate)} fps")