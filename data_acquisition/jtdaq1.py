#! /usr/bin/env python

# todo:
# advance output name
# some way to cycle output data? not saved, just background
# add color

# plots = [1, 2, 3]
# plots = [[1], [2], [3]]
# plots = [1, 2, [0j, 3]]
# plots = [[1, 'b:'], [2, 0j, 'g-']]

from sys import argv
import sys
import os
import struct
import time as t

import pygtk
import gtk
from gtk import gdk
import gobject

from pylab import *
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.font_manager
from numpy import save

import comedi as c

class Scope(gtk.Window):
    def __init__(self, plots=[1,2,4,5], rd_range=0, wr_range=0, freq=1000,
                 scantype=1, buffer=150, shown=20,
                 outchans_dotted=True):
        
        '''
        Scope is the object to initialize, display, and update several
        comedi input and output channels.

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

        scantype: the dynamic plotting type to display.  0 is a chart,
        which, like a chart recorder, keeps the drawing pen in one
        spot, but moves the drawn waveform to the left.  1 is scope
        plot, which, like an oscilloscope, draws with a moving pen, so
        the waveform stays static and the updating portion sweeps to
        the right, then wraps back to the left.

        '''


        ### set up the structure of channels and plots
        self.n_plots = len(plots)
        self.rd_chans, self.wr_chans, rd_plots, wr_plots = [],[],[],[]
        for sp in arange(self.n_plots):
            if not hasattr(plots[sp], '__iter__'): #this is not a list, so make it one
                plots[sp] = [plots[sp]]
            for ln in arange(len(plots[sp])):
                if iscomplexobj(plots[sp][ln]):
                    self.wr_chans.append(int(plots[sp][ln].imag))
                    wr_plots.append(sp)
                else:
                    self.rd_chans.append(plots[sp][ln])
                    rd_plots.append(sp)
        self.n_rd_chans = len(self.rd_chans)
        self.n_wr_chans = len(self.wr_chans)
        self.wr_range = wr_range

        # get comedi device info
        self.dev = c.comedi_open('/dev/comedi0')
        if not self.dev: raise Exception('Error opening comedi device')
            self.file_descr = c.comedi_fileno(self.dev)
        if self.file_descr<=0: raise Exception("Error obtaining Comedi device file descriptor")
        self.board_name = c.comedi_get_board_name(self.dev)
        if self.board_name == 'usbdux':
            self.rd_div = 2
            self.rd_fmt = 'H'
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
        print('x', len(self.in_range_objs), self.in_range_objs)
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
        print('rd chans', len(self.rd_chans), self.rd_chans)
        for i in range(len(self.rd_chans)):
            print('i', i, self.in_range_mins[i], self.rd_ranges[i])
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
        self.rd_cmd.scan_begin_src = c.TRIG_TIMER
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
        dump_cmd(self.rd_cmd) #print(the values)
        test_cmd(self.dev, self.rd_cmd)

        ### set up the gtk window
        gtk.Window.__init__(self)
        self.set_default_size(1000, 1000)
        self.set_title('jtdaq - %s @ %dHz'%(self.board_name, self.sample_freq))
        self.set_border_width(8)
        self.connect('destroy', lambda x: self.quitall())
        # only have to add one box to self (the main gtk window), others are added to that box
        mainvbox = gtk.VBox(False, 0)
        self.add(mainvbox)

        #set up the uigroup for menu and toolbar
        uimanager = gtk.UIManager()
        accelgroup = uimanager.get_accel_group()
        self.add_accel_group(accelgroup)
        actiongroup = gtk.ActionGroup('scope_action_group')
        ui = '''<ui>
        <menubar name="MenuBar">
          <menu action="File">
            <menuitem action="Input"/>
            <menuitem action="Output"/>
            <menuitem action="Save"/>
            <menuitem action="Quit"/>
          </menu>
          <menu action="Playmenu">
            <menuitem action="Cancel"/>
            <menuitem action="PlayRepeat"/>
            <menuitem action="Play"/>
            <menuitem action="PlayRecord"/>
          </menu>
          <menu action="Display">
            <menuitem action="Set10"/>
            <menuitem action="Set20"/>
            <menuitem action="Set30"/>
            <menuitem action="Set40"/>
            <menuitem action="Set50"/>
            <menuitem action="Set60"/>
            <menuitem action="Set70"/>
            <menuitem action="Set80"/>
            <menuitem action="Set90"/>
            <menuitem action="Set100"/>
            <menuitem action="SetMore"/>
            <menuitem action="SetLess"/>
          </menu>
        </menubar>
         <toolbar name="Toolbar">
          <toolitem action="Input"/>
          <toolitem action="Output"/>
          <toolitem action="Save"/>
          <separator/>
          <toolitem action="Cancel"/>
          <toolitem action="PlayRepeat"/>
          <toolitem action="Play"/>
          <toolitem action="PlayRecord"/>
          <separator/>
          <toolitem action="Quit"/>
        </toolbar>
        </ui>'''

        actiongroup.add_actions([
            ('File', None, '_File'),
            ('Input', gtk.STOCK_DIRECTORY, '_Save directory.', 'd',
             'Choose a save directory', self.save_data_dir),
            ('Output', gtk.STOCK_OPEN, '_Load output', 'l',
             'Load something to output', self.load_write_data_file),
            ('Save', gtk.STOCK_SAVE, '_Save output', 'space',
             'Save the current traces', self.save_current_trace),
            ('Playmenu', None, '_Playmenu'),
            ('PlayRepeat', gtk.STOCK_REFRESH, '_Play repeat', '<Alt>space',
             'Play repeating loaded output', self.write_repeat_output),
            ('Play', gtk.STOCK_MEDIA_PLAY, '_Play output', '<Control>space',
             'Play loaded output once', self.write_output),
            ('PlayRecord', gtk.STOCK_MEDIA_RECORD, '_Play output and save input', 'space',
             'Play loaded output and save input', self.write_record_output),
            ('Cancel', gtk.STOCK_MEDIA_STOP, '_Cancel output', 'BackSpace',
             'Cancel in progress output', self.write_cancel),
            ('Quit', gtk.STOCK_QUIT, '_Quit me!', None,
             'Quit', self.destroy_from_button),
            ('Display', None, '_Display'),
            ('Set10', None, '_Show 10s', '1',
             'Display to 10s', self.set_numshowing),
            ('Set20', None, '_Show 20s', '2',
             'Display to 20s', self.set_numshowing),
            ('Set30', None, '_Show 30s', '3',
             'Display to 30s', self.set_numshowing),
            ('Set40', None, '_Show 40s', '4',
             'Display to 40s', self.set_numshowing),
            ('Set50', None, '_Show 50s', '5',
             'Display to 50s', self.set_numshowing),
            ('Set60', None, '_Show 60s', '6',
             'Display to 60s', self.set_numshowing),
            ('Set70', None, '_Show 70s', '7',
             'Display to 70s', self.set_numshowing),
            ('Set80', None, '_Show 80s', '8',
             'Display to 80s', self.set_numshowing),
            ('Set90', None, '_Show 90s', '9',
             'Display to 90s', self.set_numshowing),
            ('Set100', None, '_Show 100s', '0',
             'Display to 100s', self.set_numshowing),
            ('SetMore', None, '_Show More', 'plus',
             'Display add 10s', self.set_numshowing),
            ('SetLess', None, '_Show Less', 'minus',
             'Display minus 10s', self.set_numshowing)
            ])
        
        # a = gtk.Action('Cancel', '_quit', 'Cancel the in progress output', gtk.STOCK_MEDIA_STOP)
        # a.connect('activate', self.write_cancel)
        # actiongroup.add_action_with_accel(a, '<Control>a')
        # a.set_accel_group(accelgroup)

        uimanager.insert_action_group(actiongroup, 0)
        uimanager.add_ui_from_string(ui)
        menubar = uimanager.get_widget('/MenuBar')
        mainvbox.pack_start(menubar, False)
        toolbar = uimanager.get_widget('/Toolbar')
        top_hbar = gtk.HBox(False, 0)#
        top_hbar.pack_start(toolbar, True)#
        mainvbox.pack_start(top_hbar, False)
        #mainvbox.pack_start(toolbar, False, False, 0)
 
        #hbox = gtk.HBox(False, 0) #now an hbox for the controls on top
        #mainvbox.pack_start(hbox, False, False, 0)

        #filename area
        innamehbox = gtk.HBox(False, 0)
        outnamehbox = gtk.HBox(False, 0)
        innameframe = gtk.Frame('Save input filename')
        innameframe.add(innamehbox)
        outnameframe = gtk.Frame('Load output filename')
        outnameframe.add(outnamehbox)
        #hbox.pack_start(innameframe, True, False, 0)
        #hbox.pack_start(outnameframe, True, False, 0)
        top_hbar.pack_start(innameframe, False, False, 0)
        top_hbar.pack_start(outnameframe, False, False, 0)

        self.indatelabel = gtk.Label(t.strftime('%Y_%m_%d_', t.localtime()))
        infilenumadj = gtk.Adjustment(value=0, lower=0, upper=999, step_incr=1, page_incr=10, page_size=0)
        self.infilenum = gtk.SpinButton(infilenumadj, 0.0, 0)
        self.advancing_rd_filecounter = True
        infileappendname = gtk.Label('.data')
        innamehbox.pack_start(self.indatelabel, False, False, 0)
        innamehbox.pack_start(self.infilenum, False, False, 0)
        innamehbox.pack_start(infileappendname, False, False, 0)

        self.outfilebox = gtk.Label('')
        outnamehbox.pack_start(self.outfilebox, False, False, 0)

        self.wr_filenames = []
        self.wr_data = []
        self.wr_replay_data = []
        self.rd_directory = os.getcwd() + '/data'
	
        #set up the reading buffer
        self.rd_data_size = self.sample_freq*buffer*self.n_rd_chans
        self.rd_data = zeros(self.rd_data_size)
        self.rd_ind = 0

        #set up time buffer
        if scantype==0: #chart mode
            self.tdata = arange(0, -buffer, -1./self.sample_freq)
        elif scantype==1: #scope mode
            self.tdata = arange(0, buffer, 1./self.sample_freq)
        self.numshowing = shown*self.sample_freq


        #now pack the plots into the main vbox on the bottom
        fig = Figure() 
        self.canvas = FigureCanvas(fig)
        mainvbox.pack_start(self.canvas, True, True)
        
        #the pylab subplots
        self.ax = []
        for i in arange(self.n_plots):
            sx = None #don't share x axis
            if i in rd_plots: #unless it's a read plot
                if rd_plots.index(i)>0: #and it's not the first read plot
                    sx = self.ax[rd_plots[0]] #then share with the first read plot
            self.ax.append(fig.add_subplot(self.n_plots,1,i+1,sharex=sx))


        #add an xlabel to the last plot
        self.ax[-1].set_xlabel('Time (s)')
            
        #add ylabel --- this looks bad and takes up room---leave it commented out for now
        #[ax.set_ylabel('(V)') for ax in self.ax]

        #add horizontal lines at 0
        [ax.axhline(0, color='k', ls=':') for ax in self.ax]

        #the plot lines for reading channels
        self.rd_lines, self.rd_vlines = [], []
        for i in arange(self.n_rd_chans):
            lab = 'in %d'%self.rd_chans[i]
            self.rd_lines.append(self.ax[rd_plots[i]].plot(self.tdata[:self.numshowing],
                                                           zeros(self.numshowing),
                                                           label=lab)[0])
            self.rd_vlines.append(self.ax[rd_plots[i]].axvline(0, lw=2))

        #the plot lines for the writing channels
        self.wr_lines, self.wr_vlines = [], []
        for i in arange(self.n_wr_chans):
            lab = 'out %d'%self.wr_chans[i]
            self.wr_lines.append(self.ax[wr_plots[i]].plot(self.tdata[:self.numshowing],
                                                           zeros(self.numshowing),
                                                           label=lab)[0])
            self.wr_vlines.append(self.ax[wr_plots[i]].axvline(0, lw=2))

        #adjust the time range limits for each channel
        for i in arange(self.n_plots):
            mintime = min(self.tdata[:self.numshowing])
            maxtime = max(self.tdata[:self.numshowing])
            self.ax[i].set_xlim(mintime, maxtime)

        #set up the legends for each subplot
        prop = matplotlib.font_manager.FontProperties(size=9) #smaller text
        for i in arange(self.n_plots):
            leg = self.ax[i].legend(loc='upper left', prop=prop)
            frame = leg.get_frame()
            frame.set_alpha(0.7) # make it semi-transparent

        #set the ylims
        for i in arange(len(self.rd_chans)):
            rd_rngs = self.rd_maxs[i] - self.rd_mins[i]
            rd_mins = self.rd_mins[i] - rd_rngs*.1
            rd_maxs = self.rd_maxs[i] + rd_rngs*.1
            
            self.rd_lines[i].get_axes().set_ylim(rd_mins, rd_maxs)

        #some variables
        self.xlim = self.ax[0].get_xlim()
        self.replay = False

        #pack the pylab toolbar at the bottom
        toolbar = NavigationToolbar(self.canvas, self)
        mainvbox.pack_start(toolbar, False, False)

        #pack a status text bar at the very bottom (like inkscape's)
        self.statusbar = gtk.Statusbar() #.push and .pop text messages
        mainvbox.pack_start(self.statusbar, False, True)
        self.statusbar.push(0, 'message')


        ### start the acquisition
        self.reading = True #just a flag to say it's running the input process
        self.writing = False#in it's not yet writing anything
        self.recording = False#or saving any data
        ret = c.comedi_command(self.dev, self.rd_cmd) #and the actual command
        if ret != 0: raise Exception('comedi_command failed...')


        ### set up the events to occur at keypresses and idle cycles and display
        #graphical button presses
        self.add_events(gdk.BUTTON_PRESS_MASK |
                        gdk.KEY_PRESS_MASK|
                        gdk.KEY_RELEASE_MASK)

        #timeout processes (data)
        self.update_rd_data_id = gobject.idle_add(self.update_rd_data) #get data from the buffer when idle

        #idle processes (display)
        if scantype==0: gobject.idle_add(self.update_chart) #update plots like a chart recorder
        elif scantype==1: gobject.idle_add(self.update_scope) #or plots like an oscilloscope

        # and display the window
        self.show_all()

    def destroy_from_button(self, b):
        '''This just issues the destroy command while absorbing the
        button argument.'''
        self.destroy()

    def quitall(self):
        '''Shut down all the gobject processes, comedi operations, and
        quit the gtk mainloop.'''
        self.statusbar.push(0,'quitting...')
        self.reading = False
        self.writing = False
        c.comedi_cancel(self.dev, 1)
        c.comedi_cancel(self.dev, 0)
        gtk.main_quit()

    def update_chart(self, *args):
        '''Draw the input plots with data with the current buffer
        contents from input channels---chart style .'''
        # refigure the axes if anything has changed
        xlim = self.ax[0].get_xlim()
        if self.xlim[1]!=0:
            self.ax[0].set_xlim(xmax=0.0)
        if self.xlim != xlim: #xlim has changed
            self.xlim = xlim
            self.numshowing = int((xlim[1]-xlim[0])*self.sample_freq)
                
        # plot the current input data
        for i in arange(self.n_rd_chans): #run through all the channels (don't know which we are in currently)
            cur_chan = mod(self.rd_ind - i, self.n_rd_chans) #calculate the current channel
            data_inds = arange(self.rd_ind-i, self.rd_ind-i-self.numshowing*self.n_rd_chans, -self.n_rd_chans)
            self.rd_lines[cur_chan].set_data(self.tdata[:self.numshowing], take(self.rd_data, data_inds)*self.rd_scales[cur_chan]+self.rd_translates[cur_chan])
            
        self.canvas.draw()
        return self.reading

    def update_scope(self, *args):
        '''Draw the input plots with data with the current buffer
        contents from input channels---scope style .'''
        # refigure the axes if anything has changed
        xlim = self.ax[0].get_xlim()
        if self.xlim[0]!=0:
            self.ax[0].set_xlim(xmin=0.0)
        if self.xlim != xlim: #xlim has changed
            self.xlim = xlim
            self.numshowing = int((xlim[1]-xlim[0])*self.sample_freq)

        # plot the current input data
        for i in arange(self.n_rd_chans):  #run through all the channels (don't know which we are in currently)
            cur_chan = mod(self.rd_ind - i, self.n_rd_chans)  
            c = mod(self.rd_ind/self.n_rd_chans, self.numshowing) #the current point in the window
            data_inds = arange(self.rd_ind-i-self.numshowing*self.n_rd_chans, self.rd_ind-i, self.n_rd_chans) #get the data indexes
            data_inds = data_inds[arange(len(data_inds))-c] #roll them to offset the current point
            self.rd_lines[cur_chan].set_data(self.tdata[:self.numshowing], take(self.rd_data, data_inds)*self.rd_scales[cur_chan]+self.rd_translates[cur_chan])
            self.rd_vlines[cur_chan].set_xdata([self.tdata[c],self.tdata[c]])

        self.canvas.draw()
        return self.reading

    def update_wr_plots(self, replot=False):
        '''Draw the output plots with data loaded in the buffer to
        send to the output channels.'''
        for i in arange(self.n_wr_chans):
            if self.wr_lines[i] != -1: #this channel is plotted
                if replot:
                    self.wr_lines[i].set_data(abs(self.tdata[:len(self.wr_data[i])]), self.wr_data[i])
                    wr_rngs = self.wr_maxs[i] - self.wr_mins[i]
                    wr_mins = self.wr_mins[i] - wr_rngs*.1
                    wr_maxs = self.wr_maxs[i] + wr_rngs*.1
                    self.wr_lines[i].get_axes().set_ylim(wr_mins, wr_maxs)
                    wr_tmin = self.tdata[0]
                    wr_tmax = abs(self.tdata[len(self.wr_data[0])])
                    self.wr_lines[i].get_axes().set_xlim(wr_tmin, wr_tmax)

                    self.wr_vlines[i].set_xdata(0)
                    
                elif self.writing: #currently outputting
                    t_ind = len(self.wr_data[0]) - int(self.wr_ind/(2*self.n_wr_chans))
                    self.wr_vlines[i].set_xdata(abs(self.tdata[t_ind]))
                    
        return self.writing


    def update_rd_data(self):
        '''Get data from the iocard buffer, decode from binary, add it
        to the rd_data field, then update the rd_ind index.'''
        data = os.read(self.file_descr, self.bufsize)
        # n = len(data)/2
        n = len(data)/self.rd_div
        if n == 0:
            return False
        # format = `n`+'H'
        format = `n`+self.rd_fmt
        self.rd_data[arange(self.rd_ind, self.rd_ind + n)-self.rd_data_size] = struct.unpack(format, data)
        self.rd_ind = mod(self.rd_ind + n, self.rd_data_size) #move the circular index to point to the most recent data read

        # this method is called in gobject_idle---to ensure it
        # eventually quits, this returns false when the program quits
        # and gobject removes it
        return self.reading 

    def update_wr_data(self):
        self.wr_ind = c.comedi_get_buffer_contents(self.dev, self.out_subdev)
        #self.statusbar.push(0,str(self.wr_ind))
        if self.wr_ind == 0: #writing is done
            c.comedi_cancel(self.dev, self.out_subdev) #kill writing command
            self.writing = False
            self.update_wr_plots(replot=True)
            if self.recording: #now there is data to save
                self.recording = False
                self.save_data(self.record_ind_start, self.rd_ind)
                self.statusbar.push(0,'Done writing, data saved.')
            else:
                self.statusbar.push(0,'Writing done.')

        if self.tobewritten > 0: #not done writing to the buffer
            self.tobewritten -= os.write(self.file_descr, self.wr_byte[-self.tobewritten:]) #write more, starting at the position (from the end) left to go
        return self.writing


    def save_data_dir(self, widget):
        '''Choose a directory for data saves.'''
        chooser = gtk.FileChooserDialog(title='Select a directory for saves',
                                        action=gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER,
                                        buttons=(gtk.STOCK_CANCEL,
                                                 gtk.RESPONSE_CANCEL,
                                                 gtk.STOCK_OPEN,
                                                 gtk.RESPONSE_OK))
        response = chooser.run()
        if response == gtk.RESPONSE_OK:
                    self.rd_directory = chooser.get_filenames()

        self.statusbar.push(0, 'Save directory: %s'%self.rd_directory[0])
        chooser.destroy()

    def advance_rd_filecounter(self):
        '''Increment the counter for consecutive filename generation.'''
        if self.advancing_rd_filecounter:
            self.infilenum.set_value(self.infilenum.get_value() + 1)

    def get_save_fn(self):
        '''Generate the filename for the current save.'''
        return '%s/%s%03d.npy'%(self.rd_directory,\
                                self.indatelabel.get_text(),\
                                self.infilenum.get_value_as_int())

    def save_current_trace(self, widget):
        '''Save the traces as they appear on the scope now.'''
        start_ind = mod(self.rd_ind - self.numshowing*self.n_rd_chans, self.rd_data_size)
        end_ind = self.rd_ind
        self.save_data(start_ind, end_ind)

    def save_data(self, record_ind_start, record_ind_end):
        '''Save the data between start and end indexes from the rd_data field'''
        record_ind_start -= record_ind_start%self.n_rd_chans #be sure to start with channel 0
        record_ind_end += record_ind_end%self.n_rd_chans
        rec_len = (record_ind_end - record_ind_start)%self.rd_data_size
        sdinds = mod(arange(rec_len) + record_ind_start, len(self.rd_data)) #get the inds, wrapping with mod
        sdata = self.rd_data[sdinds].reshape(rec_len/self.n_rd_chans, self.n_rd_chans).T #shape the data into array[channels x samples]
        sdata = sdata * array(self.rd_scales)[:,newaxis] + array(self.rd_translates)[:,newaxis]
        fn = self.get_save_fn()
        save(fn, sdata)
        self.advance_rd_filecounter()
        self.statusbar.push(0, 'Saved %s'%fn)

        #data_inds = arange(self.rd_ind-i, self.rd_ind-i-self.numshowing*self.n_rd_chans, -self.n_rd_chans)
        #take(self.rd_data, data_inds)*self.rd_scales[cur_chan]+self.rd_translates[cur_chan])


    def load_write_data_file(self, widget):
        chooser = gtk.FileChooserDialog(title='Load a data file for output',
                                        action=gtk.FILE_CHOOSER_ACTION_OPEN,
                                        buttons=(gtk.STOCK_CANCEL,
                                                 gtk.RESPONSE_CANCEL,
                                                 gtk.STOCK_OPEN,
                                                 gtk.RESPONSE_OK))
        chooser.set_select_multiple(True)
        response = chooser.run()
        if response == gtk.RESPONSE_OK:
                    self.wr_filenames = chooser.get_filenames()
        if self.wr_filenames != []:
            self.load_write_data()
            self.statusbar.push(0,'Data loaded.')
            self.outfilebox.set_text(self.wr_filenames[0])
            self.update_wr_plots(replot=True)
        else:
            self.statusbar.push(0,'No data loaded.')
        chooser.destroy()

    def load_write_data(self):
        '''Load a new set of waveforms assuming rows of data, with the
        first value specifying the output channel to use.'''
        old_chans, old_lines, old_vlines = self.wr_chans, self.wr_lines, self.wr_vlines #save these in case any of the new channels are plotted

        #load the data and set up the lists
        data = loadtxt(self.wr_filenames[0])
        if data.ndim==1:
            data = data.reshape(1,data.shape[-1])
        self.test_data = data
        self.wr_chans = [int(chan) for chan in data[:,0]] #the first entry in a row specifies the output channel
        self.wr_data = data[:,1:]
        self.n_wr_chans = len(self.wr_chans)
        self.wr_ranges = [self.wr_range]*self.n_wr_chans

        #this reassigns the plots, using -1 if no plot is set up for an output channel
        self.wr_lines, self.wr_vlines, self.wr_mins, self.wr_maxs = [],[],[],[]
        for i in range(len(self.wr_chans)):
            if old_chans.count(self.wr_chans[i]):
                ind = old_chans.index(self.wr_chans[i])
                self.wr_lines.append(old_lines[old_chans.index(self.wr_chans[i])])
                self.wr_vlines.append(old_vlines[old_chans.index(self.wr_chans[i])])
            else:
                self.wr_lines.append(-1)
                self.wr_vlines.append(-1)

            self.wr_mins.append(self.out_range_mins[i][self.wr_ranges[i]])
            self.wr_maxs.append(self.out_range_maxs[i][self.wr_ranges[i]])

        
    def write_cancel(self, widget):
        '''Halt the writing command in progress'''
        c.comedi_cancel(self.dev, self.out_subdev)
        self.statusbar.push(0, 'Output cancelled...')
        self.writing = False
        self.recording = False
        
    def intn_trig(self, subdev):
        '''Sets off the internal trigger on a subdevice.'''
        data = c.chanlist(1)
        data[0] = 0

        insn = c.comedi_insn_struct()
        insn.insn = c.INSN_INTTRIG
        insn.subdev = subdev
        insn.data = data
        insn.n = 1
        return c.comedi_do_insn(self.dev,insn)


    def write_record_output(self, widget=None):
        '''
        Write the output and save the resulting input channels.
        '''
        self.recording = True
        ret = self.write_output()
        if ret!=1: self.recording = False #writing failed


    def write_output(self, widget=None):
        '''
        Write the output in self.wr_data to the channels in
        self.wr_chans.  This function needs wr_data, wr_chans,
        wr_n_chans, wr_ranges, filled in.
        '''
        if self.wr_data == []:
            self.statusbar.push(0,'No data loaded...')
            return -1
        if self.writing: #still in the middle of another output command
            self.statusbar.push(0,'Still writing...')
            return -1
        # prepare the output
        wr_flat = array(self.wr_data).T.flatten().tolist() #a flat list of woven values
        wr_byte = [c.comedi_from_phys(wr_flat[i],
                                      self.out_range_objs[self.wr_chans[i%self.n_wr_chans]][self.wr_ranges[i%self.n_wr_chans]],
                                      self.out_maxdata[self.wr_chans[i%self.n_wr_chans]])
                   for i in range(len(wr_flat))] #change to physical units
        wr_byte = [struct.pack('H', item) for item in wr_byte] #change to strings of two byte numbers
        self.wr_byte = ''.join(wr_byte)  # join it all together in one string
        self.test = self.wr_byte
        #comedi object chanlist needed for the command
        self.wr_chan_obj_list = c.chanlist(self.n_wr_chans)
        for i in range(self.n_wr_chans):
            self.wr_chan_obj_list[i] = c.cr_pack(self.wr_chans[i], self.wr_ranges[i], c.AREF_GROUND)
        #the comedi command:
        self.wr_cmd = c.comedi_cmd_struct()
        self.wr_cmd.subdev = self.out_subdev              #for output
        self.wr_cmd.start_src = c.TRIG_INT                #internal trigger
        self.wr_cmd.start_arg = 0                         #delay 0ns
        self.wr_cmd.scan_begin_src = c.TRIG_TIMER         #scans start periodically
        self.wr_cmd.scan_begin_arg = self.sample_period   #ns between scanning all the channels
        self.wr_cmd.convert_src = c.TRIG_NOW              #convert immediately
        self.wr_cmd.convert_arg = 0
        self.wr_cmd.scan_end_src = c.TRIG_COUNT           #number of total channels to scan
        self.wr_cmd.scan_end_arg = self.n_wr_chans        #it's this many
        self.wr_cmd.stop_src = c.TRIG_COUNT               #number of total nums to write
        self.wr_cmd.stop_arg = len(wr_flat)               #it's this number
        self.wr_cmd.chanlist = self.wr_chan_obj_list
        self.wr_cmd.chanlist_len = self.n_wr_chans
        # where does this come from?
        dump_cmd(self.wr_cmd) #print the values
        test_cmd(self.dev, self.wr_cmd)

        #execute the output command
        ret = c.comedi_command(self.dev, self.wr_cmd)
        if ret != 0: raise Exception('comedi_command failed...')
        self.tobewritten = len(self.wr_byte) #how much total data to write
        self.tobewritten -= os.write(self.file_descr, self.wr_byte) #queue the data for output
        self.statusbar.push(0, 'tobewritten %d bytes'%self.tobewritten)
        self.writing = True
        if self.recording: self.record_ind_start = self.rd_ind #mark this point in the rd buffer
        self.intn_trig(self.out_subdev) #start the output
        self.update_wr_data_id = gobject.idle_add(self.update_wr_data) #get data from the buffer when idle
        self.update_wr_plot_id = gobject.idle_add(self.update_wr_plots) #get data from the buffer when idle
        return 1

    def write_repeat_output(self):
        '''This is to play output continuously in the background'''
        pass

    def set_numshowing(self, widget):
        name = widget.get_name()
        if name == 'Set10': self.numshowing = 10*self.sample_freq
        if name == 'Set20': self.numshowing = 20*self.sample_freq
        if name == 'Set30': self.numshowing = 30*self.sample_freq
        if name == 'Set40': self.numshowing = 40*self.sample_freq
        if name == 'Set50': self.numshowing = 50*self.sample_freq
        if name == 'Set60': self.numshowing = 60*self.sample_freq
        if name == 'Set70': self.numshowing = 70*self.sample_freq
        if name == 'Set80': self.numshowing = 80*self.sample_freq
        if name == 'Set90': self.numshowing = 90*self.sample_freq
        if name == 'Set100': self.numshowing = 100*self.sample_freq
        if name == 'SetMore': self.numshowing += 10*self.sample_freq
        if name == 'SetLess' and self.numshowing > 10*self.sample_freq:
            self.numshowing -= 10*self.sample_freq

        #adjust the time range limits for each channel
        for i in arange(self.n_plots):
            mintime = min(self.tdata[:self.numshowing])
            maxtime = max(self.tdata[:self.numshowing])
            self.ax[i].set_xlim(mintime, maxtime)


def dump_cmd(cmd):
    print("---------------------------")
    print("command structure contains:")
    print("cmd.subdev : ", cmd.subdev)
    print("cmd.flags : ", cmd.flags)
    print("cmd.start :\t", cmd.start_src, "\t", cmd.start_arg)
    print("cmd.scan_beg :\t", cmd.scan_begin_src, "\t", cmd.scan_begin_arg)
    print("cmd.convert :\t", cmd.convert_src, "\t", cmd.convert_arg)
    print("cmd.scan_end :\t", cmd.scan_end_src, "\t", cmd.scan_end_arg)
    print("cmd.stop :\t", cmd.stop_src, "\t", cmd.stop_arg)
    print("cmd.chanlist : ", cmd.chanlist)
    print("cmd.chanlist_len : ", cmd.chanlist_len)
    print("cmd.data : ", cmd.data)
    print("cmd.data_len : ", cmd.data_len)
    print("---------------------------")


def cmdtest_messages(index):
    return ["success",
            "invalid source",
            "source conflict",
            "invalid argument",
            "argument conflict",
            "invalid chanlist"][index]

def test_cmd(dev, cmd):
    ret = c.comedi_command_test(dev, cmd)
    print('cmd test returns %d, %s'%(ret, cmdtest_messages(ret)))
    if ret<0:
        raise Exception('comedi_command_test failed')


# Notes
#it used to be that period meant time between sampling single channels,
#now it seems to mean between samples of all channels (more intuitive, but
#different from my old code).
#self.sample_period = 1000000000/(self.sample_freq*self.nchans) #period in nanoseconds
#now:
#self.sample_period = 1000000000/self.sample_freq #period in nanoseconds



if len(argv) > 1: argv_range_n = int(argv[1])
else: channel_ranges = 1

if len(argv) > 2: argv_n_chans = int(argv[2])
else: argv_n_chans = 4


# chanlist inputs [1,2,4,5,6,7] #skips thermocouple inputs 0 and 3
# argv_chanlist = [0,0j,1j,2j,3j]
argv_chanlist = [1,2,4,5,6,7]
argv_rd_range = [2,2,2,2,2,2]

s = Scope(argv_chanlist, rd_range=argv_rd_range, scantype=0, shown=120)

gtk.main()

