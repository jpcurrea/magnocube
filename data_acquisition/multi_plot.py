import time
import threading
from threading import Timer
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg   
from collections import deque
import sys
import cProfile


SAMPLE_RATE = 1000.              # rate of DAQ sampling in Hz
BUFFER_SIZE = round(10 * SAMPLE_RATE)             # number of frames to store in buffer
DISPLAY_RATE = 12               # update rate in Hz

# for each channel, plot a line graph
channels = [
    'exp_indicator', 'test_indicator', 'projector_frame', 'camera_frame', 'variable1']

buffer_arr = np.zeros((len(channels), BUFFER_SIZE), dtype=float)


# make a worker thread to update the buffer array in parallel
class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

# make update function to be called by a timer
class Plotter():
    def __init__(self, channels=channels):
        # setup the buffer array used as a circular buffer
        self.buffer_arr = np.zeros((len(channels), BUFFER_SIZE), dtype=float)
        # todo: use deque instead of numpy array to speed things up: not better
        # self.buffer_arr = [deque([], maxlen=BUFFER_SIZE) for chanell in channels]
        # store the channels and index information
        self.channels = channels
        self.circular_index = 0
        self.order = np.arange(BUFFER_SIZE)
        self.circular_indices = np.arange(BUFFER_SIZE)
        # get the time value for each frame
        self.times = np.linspace(-BUFFER_SIZE/SAMPLE_RATE, 0, BUFFER_SIZE)
        # initialize the application, window, and plot
        self.app = QtGui.QApplication([])                                                        
        self.view = pg.GraphicsView()                                                            
        self.layout = pg.GraphicsLayout()
        self.view.setCentralItem(self.layout)
        self.view.show()                                                                         
        self.view.resize(600, len(channels) * 200)
        # add plots for each channel
        self.plots = []
        first_plot = None       # use first plot to link all the x-axes
        for num, (channel, vals) in enumerate(zip(channels, buffer_arr)):
            plot = pg.PlotWidget()
            plot = self.layout.addPlot(num, 0, x=self.times[::10], y=vals[::10], name=channel, title=channel)
            plot.showGrid(x=True, y=True, alpha=.5)
            self.plots += [plot]
            if first_plot is None:
                first_plot = plot
            else:
                plot.setXLink(first_plot)
            old_plot = plot
        # pre-process random values to avoid calling the random function too often
        self.new_vals = np.random.random(BUFFER_SIZE)

    def update_val_timer(self):
        while True:
            self.update_val()
            time.sleep(1./SAMPLE_RATE)
        # start a QTimer to call self.update_val repeatedly
        # self.timer = QtCore.QTimer()     
        # self.timer.timeout.connect(self.update_val)
        # self.timer.start(1000./SAMPLE_RATE)
        # use Python threading Timer instead
        # self.timer = RepeatedTimer(1000./SAMPLE_RATE, self.update_val)

    def update_val(self):
        new_vals = self.new_vals[self.circular_index]
        # new_vals = np.random.random(len(self.channels))
        # for val, buffer_channel in zip(new_vals, self.buffer_arr):
        #     buffer_channel.append(val)
        self.buffer_arr[:, self.circular_index] = new_vals
        # update the circular index
        # self.order += 1
        # self.order %= BUFFER_SIZE - 1
        self.circular_index += 1
        self.circular_index %= BUFFER_SIZE - 1
        # self.update_plot()

    def update_val_thread(self):
        # make a thread to call update_val_timer independently
        self.thread = threading.Thread(target=self.update_val_timer)
        self.thread.start()
        # 1. make a QThread
        # self.thread = QtCore.QThread()
        # self.thread.__init__()
        # # 2. make a worker
        # self.worker = Worker(plotter=self)
        # # 3. move worker into the thread
        # self.worker.moveToThread(self.thread)
        # # 4. connect signals and slots
        # self.thread.started.connect(self.worker.run)
        # self.worker.finished.connect(self.thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        # self.thread.finished.connect(self.thread.deleteLater)
        # self.worker.progress.connect(self.reportProgress)
        # # Step 6: Start the thread
        # self.thread.start()
        

    def update_plot(self):
        # update the plots by adding moving the current frame index 
        for num, (plot, vals) in enumerate(zip(self.plots, self.buffer_arr)):
            ordered_vals = np.append(vals[self.circular_index:],
                                     vals[:self.circular_index])
            plot.dataItems[0].setData(x=self.times[::10],
                                      y=ordered_vals[::10])


profiler = cProfile.Profile()
profiler.enable()

    
plotter = Plotter(channels=channels)
# plotter.update_val_thread()
# setup timers to update the values and the plot separately
# to update values:
timer_val = QtCore.QTimer()     
timer_val.timeout.connect(plotter.update_val)
timer_plot = QtCore.QTimer()    # update the plot
timer_plot.timeout.connect(plotter.update_plot)
# start the timers
timer_val.start(1000./SAMPLE_RATE)
timer_plot.start(1000./DISPLAY_RATE)

plotter.app.exec()
# if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):         
#     QtGui.QApplication.instance().exec_()  

profiler.disable()
profiler.print_stats(sort='pcall')
profiler.print_stats(sort='ncalls')
profiler.print_stats(sort='cumtime')

