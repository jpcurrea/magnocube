# -*- coding: utf-8 -*-

from sys import argv
import sys
import os
import struct
import time as t

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import comedi as c

qt_app = QtGui.QApplication(sys.argv)

class Scope(QtGui.QMainWindow):
    def __init__(self, plots=[1,2,4,5], rd_range=0, freq=1000,
                 buffer=2400, shown=20):
        
        '''
        Scope is the object to initialize, display, and update several
        comedi input channels.

        arguments:
        
        plots: a list describing the subplotting display. Optional
        sublists group multiple plots on one axis, real integers (n)
        specify input channels, and imaginary integers (nj) specify
        output channels.  plots=[0,1,2,3] displays 4 rows of plots,
        each showing a different input channel, while
        plots=[0,[1,3],[0,0j,1j]] displays 3 rows of plots, the top
        showing input channel 0, the next showing input channels 1 and
        3, and the last showing input channel 0 and output channels 0
        and 1.

        freq: the sampling frequency in Hz, used for both input and
        output.  These could be two seperate frequencies, but I have
        no current need for this, so to keep the code simple, for now
        there is just one.
        '''

        super(Scope, self).__init__()
        self.setWindowTitle('jtdaq scope')
        self.setMinimumWidth(1600)
        self.setMinimumHeight(1000)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        ### set up the structure of channels and plots
        self.num_plots = len(plots)

        ### get comedi device info
        self.dev = c.comedi_open('/dev/comedi0')
        if not self.dev:
            # raise Exception('Error opening comedi device')
            print('Error opening comedi device')
            self.read = self.read_dummy
        else:
            self.file_descr = c.comedi_fileno(self.dev)
            if self.file_descr<=0: raise Exception("Error obtaining Comedi device file descriptor")
            self.board_name = c.comedi_get_board_name(self.dev)
            if self.board_name == 'usbdux':
                self.rd_div = 2
                self.rd_fmt = 'H'
                # self.rd_fmt = repr(n)
            elif self.board_name == 'usbduxsigma':
                self.rd_div = 4
                self.rd_fmt = 'I'

            self.n_subdevs = c.comedi_get_n_subdevices(self.dev) 

            self.in_subdev = c.comedi_get_read_subdevice(self.dev) #streaming input subdev number
            self.out_subdev = c.comedi_get_write_subdevice(self.dev) #streaming output subdev number
            self.dio_subdev = None #dio subdev, requires looking
            for i in range(self.n_subdevs): 
                if c.comedi_get_subdevice_type(self.dev, i) == 5:
                    self.dio_subdev = i

            self.n_in_chans = c.comedi_get_n_channels(self.dev, self.in_subdev)
            self.n_out_chans = c.comedi_get_n_channels(self.dev, self.out_subdev)

            #maxdata is the maximal integer output or input (comedi units) for a channel
            self.in_maxdata = [c.comedi_get_maxdata(self.dev, self.in_subdev, chan)\
                               for chan in range(self.n_in_chans)]
            self.out_maxdata = [c.comedi_get_maxdata(self.dev, self.out_subdev, chan)\
                                for chan in range(self.n_out_chans)]
            #ranges are in physical units (such as voltages) for each channel
            self.n_in_ranges = [c.comedi_get_n_ranges(self.dev, self.in_subdev, chan)\
                                for chan in range(self.n_in_chans)]
            self.n_out_ranges = [c.comedi_get_n_ranges(self.dev, self.out_subdev, chan)\
                                 for chan in range(self.n_out_chans)]
            #range objects in these 2d lists have fields like min and max, for the voltage range in or out
            self.in_range_objs = [[c.comedi_get_range(self.dev, self.in_subdev, chan, rng)\
                               for rng in range(self.n_in_ranges[chan])]\
                              for chan in range(self.n_in_chans)] #reference by self.in_ranges[channel][range]
            self.out_range_objs = [[c.comedi_get_range(self.dev, self.out_subdev, chan, rng)\
                                for rng in range(self.n_out_ranges[chan])]\
                               for chan in range(self.n_out_chans)] #reference by self.out_ranges[channel][range]
            print ('x', len(self.in_range_objs), self.in_range_objs)
            self.in_range_mins = [[a.min for a in b] for b in self.in_range_objs]
            self.in_range_maxs = [[a.max for a in b] for b in self.in_range_objs]
            self.in_range_rngs = [[a.max-a.min for a in b] for b in self.in_range_objs]
            self.out_range_mins = [[a.min for a in b] for b in self.out_range_objs]
            self.out_range_maxs = [[a.max for a in b] for b in self.out_range_objs]
            self.out_range_rngs = [[a.max-a.min for a in b] for b in self.out_range_objs]

            # setup comedi input command
            self.sample_freq = freq
            #it used to be that period meant time between sampling a single channel,
            #now it seems to mean between samples of all channels (more intuitive, but different).
            self.sample_period = 1000000000/self.sample_freq #period in nanoseconds
            self.bufsize = 10000
            # the ranges
            if hasattr(rd_range, '__iter__'): self.rd_ranges = rd_range
            else: self.rd_ranges = [rd_range]*self.n_rd_chans
            # get these for transforming later
            self.rd_scales, self.rd_translates, self.rd_mins, self.rd_maxs = [],[],[],[]
            print ('rd chans', len(self.rd_chans), self.rd_chans)
            for i in range(len(self.rd_chans)):
                print ('i', i, self.in_range_mins[i], self.rd_ranges[i])
                self.rd_mins.append(self.in_range_mins[i][self.rd_ranges[i]])
                self.rd_maxs.append(self.in_range_maxs[i][self.rd_ranges[i]])
                self.rd_scales.append(self.in_range_rngs[i][self.rd_ranges[i]]/self.in_maxdata[i]) #range/maxdata
                self.rd_translates.append(self.in_range_mins[i][self.rd_ranges[i]])

            # the object chanlist
            self.rd_chan_obj_list = c.chanlist(self.n_rd_chans)
            for i in range(self.n_rd_chans):
                self.rd_chan_obj_list[i] = c.cr_pack(self.rd_chans[i], self.rd_ranges[i], c.AREF_GROUND)
            #the comedi command:
            self.rd_cmd = c.comedi_cmd_struct()
            self.rd_cmd.start_src = c.TRIG_NOW
            self.rd_cmd.start_arg = 0
            self.rd_cmd.scan_begin_src = c.TRIG_Timer
            self.rd_cmd.scan_begin_arg = self.sample_period
            self.rd_cmd.convert_src = c.TRIG_NOW
            self.rd_cmd.convert_arg = 0
            self.rd_cmd.scan_end_src = c.TRIG_COUNT
            self.rd_cmd.scan_end_arg = self.n_rd_chans
            self.rd_cmd.stop_src = c.TRIG_NONE
            self.rd_cmd.stop_arg = 0
            self.rd_cmd.chanlist = self.rd_chan_obj_list
            self.rd_cmd.chanlist_len = self.n_rd_chans
            #check the acquisition command
            dump_cmd(self.rd_cmd) #print the values
            test_cmd(self.dev, self.rd_cmd)


        # Create the QVBoxLayout that lays out the whole form
        w = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout(w)
        self.setCentralWidget(w)

    def read_dummy(self):
        return np.random.randn()



# # 1) Simplest approach -- update data in the array such that plot appears to scroll
# #    In these examples, the array size is fixed.
# p1 = win.addPlot()
# p2 = win.addPlot()
# data1 = np.random.normal(size=300)
# curve1 = p1.plot(data1)
# curve2 = p2.plot(data1)
# ptr1 = 0
# def update1():
#     global data1, curve1, ptr1
#     data1[:-1] = data1[1:]  # shift data in the array one sample left
#                             # (see also: np.roll)
#     data1[-1] = np.random.normal()
#     curve1.setData(data1)
    
#     ptr1 += 1
#     curve2.setData(data1)
#     curve2.setPos(ptr1, 0)
    

# # 2) Allow data to accumulate. In these examples, the array doubles in length
# #    whenever it is full. 
# win.nextRow()
# p3 = win.addPlot()
# p4 = win.addPlot()
# # Use automatic downsampling and clipping to reduce the drawing load
# p3.setDownsampling(mode='peak')
# p4.setDownsampling(mode='peak')
# p3.setClipToView(True)
# p4.setClipToView(True)
# p3.setRange(xRange=[-100, 0])
# p3.setLimits(xMax=0)
# curve3 = p3.plot()
# curve4 = p4.plot()

# data3 = np.empty(100)
# ptr3 = 0

# def update2():
#     global data3, ptr3
#     data3[ptr3] = np.random.normal()
#     ptr3 += 1
#     if ptr3 >= data3.shape[0]:
#         tmp = data3
#         data3 = np.empty(data3.shape[0] * 2)
#         data3[:tmp.shape[0]] = tmp
#     curve3.setData(data3[:ptr3])
#     curve3.setPos(-ptr3, 0)
#     curve4.setData(data3[:ptr3])


# # 3) Plot in chunks, adding one new plot curve for every 100 samples
# chunkSize = 100
# # Remove chunks after we have 10
# maxChunks = 10
# startTime = pg.ptime.time()
# win.nextRow()
# p5 = win.addPlot(colspan=2)
# p5.setLabel('bottom', 'Time', 's')
# p5.setXRange(-10, 0)
# curves = []
# data5 = np.empty((chunkSize+1,2))
# ptr5 = 0

# def update3():
#     global p5, data5, ptr5, curves
#     now = pg.ptime.time()
#     for c in curves:
#         c.setPos(-(now-startTime), 0)
    
#     i = ptr5 % chunkSize
#     if i == 0:
#         curve = p5.plot()
#         curves.append(curve)
#         last = data5[-1]
#         data5 = np.empty((chunkSize+1,2))        
#         data5[0] = last
#         while len(curves) > maxChunks:
#             c = curves.pop(0)
#             p5.removeItem(c)
#     else:
#         curve = curves[-1]
#     data5[i+1,0] = now - startTime
#     data5[i+1,1] = np.random.normal()
#     curve.setData(x=data5[:i+2, 0], y=data5[:i+2, 1])
#     ptr5 += 1


# # update all plots
# def update():
#     update1()
#     update2()
#     update3()
# timer = pg.QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(50)

s = Scope()
s.run()


## Start Qt event loop unless running in interactive mode or using pyside.
# if __name__ == '__main__':
#     import sys
#     if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
#         print ('run scope')
#         s = Scope()
#         s.run()
