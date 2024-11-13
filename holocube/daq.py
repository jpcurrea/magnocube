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

class DAQ():
    def __init__(self, config_fn="./DAQ.ini", storage_directory="./", show=True):
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
        # setup the analog channels
        # get the list of digital and analog channels to record from
        self.inputs_digital = dict(config['inputs_digital'])
        self.inputs_analog = dict(config['inputs_analog'])
        self.buffer_analog, self.buffer_digital = [], []
        # set the sample rate and bin size for rendering at the desired display rate
        self.bin_size = round(float(self.sample_rate)/float(self.display_rate))
        # self.bin_size = 1
        self.input_tasks = []
        for input_type, channels in zip(['analog', 'digital'], [self.inputs_analog, self.inputs_digital]):
            if len(channels) > 0:
                task = nidaqmx.Task()
                self.__setattr__(f"task_{input_type}", task)
                self.input_tasks += [task]
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
        # do the same, setting up stream writers for each output channel
        self.outputs_analog = dict(config['outputs_analog'])
        self.outputs_digital = dict(config['outputs_digital'])
        self.output_tasks = []
        self.digital_writers = {}
        self.analog_writers = {}
        for output_type, channels in zip(['analog', 'digital'], [self.outputs_analog, self.outputs_digital]):
            if len(channels) > 0:
                task = nidaqmx.Task()
                self.__setattr__(f"task_{output_type}_out", task)
                self.output_tasks += [task]
                for name, val in channels.items():
                    if output_type == 'analog':
                        task.ao_channels.add_ao_voltage_chan(val, min_val=-5, max_val=5)
                        writer = stream_writers.AnalogSingleChannelWriter(task.out_stream)
                        self.analog_writers[name] = writer
                    else:
                        task.do_channels.add_do_chan(val, line_grouping=nidaqmx.constants.LineGrouping.CHAN_FOR_ALL_LINES)
                        writer = stream_writers.DigitalSingleChannelWriter(task.out_stream)
                        self.digital_writers[name] = writer        # plot the channels
        # setup the plot
        self.scale = max(round(float(self.buffer_size) / 10000.), 1)
        print(f"bin size = {self.bin_size}")
        # start a circular indexing
        self.current_frame_analog = 0
        self.current_frame_digital = 0
        if show:
            # plot the figure
            self.plot()
            # setup autoscaling whenever the user changes the x-range
            # self.plots[0].sigXRangeChanged.connect(self.set_scale)
        # low_time is the duration of off and high_time is the duration of on signal
        # start a PWM signal using ctr0, which outputs to PFI12 based on NiMAX
        # add callback to the task
        self.is_reading = False
        self.recording = False
        self.start_readers()
        if show:
            # make a QTimer to update the plot
            self.plotter_thread()

    def start_readers(self):
        """Make a separate thread to repeatedly run self.read()."""
        # set is_reading to true 
        self.is_reading = True
        # make the thread for reading analog data 
        if len(self.inputs_analog) > 0:
            self.analog_reader = threading.Thread(target=self.reader_analog, name='reader_analog')
            self.analog_reader.start()
        # make the thread for reading digital data 
        if len(self.inputs_digital) > 0:
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
            # store the temporary buffer into the h5 file if recording
            if self.recording:
                self.recorded_frames += 1
                self.record_h5['digital'].resize((len(self.inputs_digital), self.recorded_frames * self.bin_size))
                self.record_h5['digital'][:, -self.bin_size:] = self.temp_buffer_digital[:]
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
                    self.buffer_analog[:, self.current_frame_analog:end_frame] = self.temp_buffer_analog[:]
            # store the temporary buffer into the h5 file if recording
            if self.recording:
                self.recorded_frames += 1
                self.record_h5['analog'].resize((len(self.inputs_analog), self.recorded_frames * self.bin_size))
                self.record_h5['analog'][:, -self.bin_size:] = self.temp_buffer_analog[:]
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

    def write_to_channel(self, channel_name, value, output_type='analog'):
        """Write a single value to the specified output channel."""
        if output_type == 'analog':
            writer = self.analog_writers.get(channel_name)
            if writer is None:
                raise ValueError(f"No writer found for analog channel '{channel_name}'")
            writer.write_one_sample(value)
        else:
            writer = self.digital_writers.get(channel_name)
            if writer is None:
                raise ValueError(f"No writer found for digital channel '{channel_name}'")
            writer.write_one_sample_port_byte(value)
            # print(f"writing {value} to {channel_name}")

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
        self.view.resize(800, len(self.inputs_analog) * 200)
        # add plots for each channel
        self.plots = []
        first_plot = None  # use first plot to link all the x-axes
        num = 0
        for channels, buffer in zip(
            [self.inputs_analog, self.inputs_digital], 
            [self.buffer_analog, self.buffer_digital]):
            for channel, vals in zip(channels.keys(), buffer):
                plot = self.layout.addPlot(row=num, col=0, x=self.times, y=vals,
                    name=channel, title=channel.replace("_", " "), colspan=4)
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
        button_layout = self.layout.addLayout(row=0, col=4, rowspan=num, colspan=1)
        buttons = [
            ('capture', self.capture),
            ('capture +\npreview', self.preview),
            ('start recording', self.start_recording_button),
            ('stop recording', self.stop_recording)
        ]

        for label, callback in buttons:
            button_proxy = QtWidgets.QGraphicsProxyWidget()
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            button_proxy.setWidget(button)
            button_layout.addItem(button_proxy)
            button_layout.nextRow()

        # add a button for writing to the output channels and a slider for the value
        for output_type, channels in zip(['analog', 'digital'], [self.outputs_analog, self.outputs_digital]):
            for channel in channels.keys():
                # Create a button for writing to the channel
                button_proxy = QtWidgets.QGraphicsProxyWidget()
                button = QtWidgets.QPushButton(f"write to {channel}")
                button.clicked.connect(partial(self.write_to_channel, channel, output_type=output_type))
                button_proxy.setWidget(button)
                button_layout.addItem(button_proxy)
                button_layout.nextRow()

                # Create a slider for setting the output value
                slider_proxy = QtWidgets.QGraphicsProxyWidget()
                slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                if output_type == 'analog':
                    slider.setMinimum(-5 if output_type == 'analog' else 0)
                    slider.setMaximum(5 if output_type == 'analog' else 255)
                else:
                    slider.setMinimum(0)
                    slider.setMaximum(255)
                    slider.setTickInterval(255)
                slider.setValue(0)
                slider.valueChanged.connect(partial(self.write_to_channel, channel, output_type=output_type))
                slider_proxy.setWidget(slider)
                button_layout.addItem(slider_proxy)
                button_layout.nextRow()

        # Set the column stretch factors
        self.layout.layout.setColumnStretchFactor(0, 4)  # Plots column
        self.layout.layout.setColumnStretchFactor(4, 1)  # Buttons column

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
        for channels, buffer in zip([self.inputs_analog, self.inputs_digital], [self.buffer_analog, self.buffer_digital]):
            for (name, port), vals in zip(
                    channels.items(), buffer):
                h5_file[name] = vals
        # and the time marks
        h5_file['time'] = self.times
        # store channel info
        h5_file['analog_info'] = tuple(self.inputs_analog)
        h5_file['digital_info'] = tuple(self.inputs_digital)
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
        for num, (buffer, channels) in enumerate(zip([buffer_analog, buffer_digital], [self.inputs_analog, self.inputs_digital])):
            for vals, channel in zip(buffer, channels):
                ax = axes[num]
                ax.plot(self.times, vals)
                ax.set_ylabel(channel.replace("_", ""))
                num += 1
        # format and plot it
        plt.tight_layout()
        plt.show()

    def start_recording_button(self):
        self.start_recording()

    def start_recording(self, h5_file=None, fn=None):
        """Open an H5 file and start storing the read data.

        Note: This works by setting self.recording to True and then
        storing the temporary buffer into both the circular buffer and
        the h5 file. 

        Parameters
        ----------
        :param fn: str default=None
            Filename to store the data in. H5 files allow for easy storage of
            large datasets without keeping it all in RAM.
        :param h5_file: h5py.File default=None
        """
        self.new_file = False
        if h5_file is not None:
            if callable(h5_file):
                h5_file = h5_file()
            self.record_h5 = h5_file
            self.record_fn = self.record_h5.filename
        elif fn is not None:
            self.record_fn = fn
            self.record_h5 = h5py.File(self.record_fn, 'w')
            self.new_file = True
        else:            
            now = datetime.now()
            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
            self.record_fn = os.path.join(self.storage_directory, timestamp + ".hdf5")
            # first, make a new file
            self.record_h5 = h5py.File(self.record_fn, 'w')
            self.new_file = True
        if self.new_file:
            print(f"recording to {self.record_fn}")
        # this should have two datasets, one for analog and one for digital
        if len(self.inputs_analog) > 0:
            self.record_h5.create_dataset(
                'analog', shape=(len(self.inputs_analog), 0), 
                maxshape=(len(self.inputs_analog), None), 
                chunks=(len(self.inputs_analog), self.bin_size))
        if len(self.inputs_digital) > 0:
            self.record_h5.create_dataset(
                'digital', shape=(len(self.inputs_digital), 0), 
                maxshape=(len(self.inputs_digital), None), 
                chunks=(len(self.inputs_digital), self.bin_size))
        # then set self.recording to True
        self.recording = True
        self.recorded_frames = 0

    def stop_recording(self):
        """Stop recording and close the h5 file."""
        self.recording = False
        if self.new_file:
            self.record_h5.close()
            # notify the user
            print("recording stopped")

    def close(self):
        """Close the DAQ and stop all processes."""
        self.is_reading = False
        # stop all running threads
        if 'analog_reader' in dir(self):
            self.analog_reader.join()
        if 'digital_reader' in dir(self):
            self.digital_reader.join()
        # close all tasks
        for task in self.tasks:
            task.close()
        # if plotting, stop the plot timer
        if 'plot_timer' in dir(self):
            self.plot_timer.stop()
            self.app.quit()


if __name__ == "__main__":
    daq = DAQ(show=True)
    daq.app.exec_()
    # # test recording and stopping without plotting
    # daq.start_recording()
    # time.sleep(5)
    # daq.stop_recording()
    # # try loading the file and plotting it using matplotlib
    # h5 = h5py.File(daq.record_fn, 'r')
    # # grab the values from the buffer
    # buffer_analog = np.copy(h5['analog'])
    # plt.plot(buffer_analog[0])
    # plt.show()
    # daq.close()