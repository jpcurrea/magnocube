from functools import partial
import configparser
import numpy as np
cupy_loaded = False
try:
    import cupy as cp
    cupy_loaded = True
except:
    cp = np
import os
import pickle
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import threading
import matplotlib
from matplotlib import pyplot as plt
import skvideo
import struct
import time
from holocube import camera
import json

blue, green, yellow, orange, red, purple = [
    (0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (0.83, 0.74, 0.37),
    (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]

class Slider(QtWidgets.QWidget):
    def __init__(self, minimum, maximum, parent=None, var_lbl="slider"):
        super(Slider, self).__init__(parent=parent)
        self.var_lbl = var_lbl
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        spacerItem = QtWidgets.QSpacerItem(
            0, 20, QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QtWidgets.QSlider(self)
        self.slider.setOrientation(QtCore.Qt.Vertical)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QtWidgets.QSpacerItem(
            0, 20, QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())
        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(abs_value=self.minimum)

    def setLabelValue(self, value=None, abs_value=None):
        if abs_value is not None:
            self.x = int(round(abs_value))
            # calculate the relative position for the slider
            pos = ((self.slider.maximum() - self.slider.minimum())
                   * (abs_value - self.minimum)
                   / (self.maximum - self.minimum))
            self.slider.setValue(int(round(pos)))
        else:
            self.x = self.minimum + (
                    (float(value) / (self.slider.maximum() - self.slider.minimum()))
                    * (self.maximum - self.minimum))
            self.x = int(round(self.x))
        self.label.setText(f"{self.var_lbl}\n{self.x:d}", )


class Marker():
    def __init__(self, name, coord, color, menu_layout, window, gui, size=10):
        """A 2D marker and the corresponding radio button.
        
        Parameters
        ----------
        name : str
            The name of the marker.
        coord : tuple
            The x, y coordinates of the marker.
        color : tuple
            The RGBA color of the marker.
        menu_layout : QtWidgets layout
            The layout to add the marker to.
        window : pyqtgraph window
            The window to add the marker to.
        gui : VideoGUI
            The parent GUI.
        """
        # store attributes
        self.size = size
        name = name.replace("_", " ")
        self.name, self.coord, self.color = name, coord, color
        self.menu, self.window = menu_layout, window
        self.gui = gui
        self.line_pen = None
        # make a horizontal sublayout
        self.layout = QtWidgets.QHBoxLayout()
        # plot the marker
        # self.marker = QtWidgets.QGraphicsEllipseItem(
        #     coord[0] - size/2, coord[1]-size/2, size, size)
        self.marker = QtWidgets.QGraphicsEllipseItem(
            0, 0, size, size)
        # self.marker.setPos(coord[0], coord[1])
        self.standby_pen = pg.mkPen(
            255 * color[0], 255 * color[1], 255 * color[2], 255,
            width=3)
        self.active_pen = pg.mkPen(
            255 * color[0], 255 * color[1], 255 * color[2], 255,
            width=6)
        self.marker.setPen(self.standby_pen)
        self.marker.setBrush(
            pg.mkBrush(255 * color[0], 255 * color[1], 255 * color[2], 255))
        self.window.addItem(self.marker)
        # add the button to the menu
        self.selected = False
        self.button = QtWidgets.QRadioButton(self.name)
        self.button.toggled.connect(self.toggle)
        # color_str = f"rgb({255*int(color[0]):.1f}, {255*int(color[1]):.1f}, {255*int(color[2]):.1f})"
        # self.button.setStyleSheet("QRadioButton::indicator"
        #                           "{"
        #                           f"background-color : rgb({255*int(color[0]):.1f}, {255*int(color[1]):.1f}, {255*int(color[2]):.1f})"
        #                           "}")
        self.button.setStyleSheet(f"color: white")
        # add the button to the layout
        self.layout.addWidget(self.button, 1)
        # add a QDoubleSpinBox for the x and y coordinates
        # x coord
        self.x_box = QtWidgets.QDoubleSpinBox()
        self.x_box.setMinimum(-2000)
        self.x_box.setMaximum(2000)
        self.x_box.setValue(self.coord[0])
        self.x_box.setStyleSheet("color: white; background-color: black")
        self.x_box.setDecimals(1)
        self.x_box.valueChanged.connect(self.update_coord)
        # y coord
        self.y_box = QtWidgets.QDoubleSpinBox()
        self.y_box.setMinimum(-2000)
        self.y_box.setMaximum(2000)
        self.y_box.setValue(self.coord[1])
        self.y_box.setStyleSheet("color: white; background-color: black")
        self.y_box.setDecimals(1)
        self.y_box.valueChanged.connect(self.update_coord)
        # add the spin boxes to the layout
        self.layout.addWidget(self.x_box, 1)
        self.layout.addWidget(self.y_box, 1)
        # add the layout to the menu
        self.menu.addLayout(self.layout, 1)
        # update the coords
        self.update_coord()

    def update_coord(self):
        """Update the coordinate of the marker."""
        coord = [self.x_box.value(), self.y_box.value()]
        # store the new coordinate
        self.coord[:] = coord
        # update the marker coordinate
        self.marker.setPos(int(coord[0] - self.size/2), int(coord[1] - self.size/2))
        # update the GUI
        self.gui.update_plots()
        self.gui.save_vars()

    def toggle(self):
        # if checked, use the large marker
        if self.button.isChecked():
            self.selected = True
            self.marker.setPen(self.active_pen)
        # otherwise, use the small marker
        else:
            self.selected = False
            self.marker.setPen(self.standby_pen)

# a custom ImageItem class to allow for marker selection
class MarkedImageItem(pg.ImageItem):
    def __init__(self, gui, **image_item_kwargs):
        self.gui = gui
        super().__init__(**image_item_kwargs)

    def mouseClickEvent(self, event):
        coord = np.array(event.pos())
        # center the coords using the center of mass
        dy, dx = -self.gui.img_height/2 - self.gui.border_pad, -self.gui.img_width/2 - self.gui.border_pad
        coord += np.array([dx, dy])
        print(coord)
        # go through all of the markers and update the coordinates of the active one
        for lbl, marker in self.gui.markers.items():
            if marker.selected:
                self.gui.marker_coords[lbl] = coord
                self.gui.save_vars()
                marker.update_coord(coord)
        


class VideoGUI(QtWidgets.QMainWindow):
    def __init__(self, img_height=600, img_width=500, display_rate=30,
                 inner_radius=100., outer_radius=150., thresh=100,
                 thresholding=True, config_fn="video_player.config", border_pad=0,
                 wingbeat_analysis=False):
        """A GUI for displaying the live video feed and heading data.


        Parameters
        ----------
        img_height, img_width : int
            The height or width of the incoming video feed.
        display_rate : int
            The rate at which to display the video feed.
        inner_radius, outer_radius : float
            The inner and outer radius intersecting with the head and abdomen 
            of the fly, respectively.
        thresh : int
            The threshold for the ring detectors.
        thresholding : bool
            Whether to threshold the incoming video feed.
        config_fn : str
            The path to the config file.
        border_pad : int
            The number of pixels to pad the image with.
        wingbeat_analysis : bool
            Whether to perform wingbeat analysis.
        heading_pad : float
            The maximum and minimum relative heading value.
        """
        self.wingbeat_analysis = wingbeat_analysis
        self.display_rate = display_rate
        self.config_fn = config_fn
        self.experiment_parameters = {}
        self.data = {}
        self.markers = {}
        self.wing_edges = {}
        self.start_time = time.time()
        self.current_time = time.time()
        params = ['heading', 'heading_smooth', 'com_shift']
        for param in params:
            self.__setattr__(param, None)
            self.data[param] = []
        # grab parameters
        self.img_height, self.img_width = img_height, img_width
        self.border_pad = border_pad
        self.thresh = thresh
        self.thresholding = thresholding
        self.inner_radius, self.outer_radius = inner_radius, outer_radius
        self.tracking_active = False
        self.import_config()
        self.config_update_time = time.time()
        super().__init__()
        # set the title
        self.setWindowTitle("Live Feed")
        # set the geometry
        # self.setGeometry(100, 100, img_width, img_height)
        # setup the components
        self.setup()
        # disable the autoRange for the heading plot
        self.heading_plot.enableAutoRange(False)
        # show all the widgets
        self.show()
        # update the plots
        self.update_plots()

    def import_config(self):
        """Import important parameters from the config file."""
        # read/make config file
        if os.path.exists(self.config_fn):
            # load using config parser
            config = configparser.ConfigParser()
            config.read(self.config_fn)
            # self.genotype_list = config.getList('experiment_parameters','genotype')
            # check if parameters stored
            keys = ['inner_r', 'outer_r', 'wing_r', 'thresh', 'invert', 'rotate270', 'flipped', 'kalman_jerk_std', 'kalman_noise', 'camera_fps', 'time_scale', 'heading_pad']
            vars = ['inner_radius', 'outer_radius', 'wing_radius', 'thresh', 'invert', 'rotate270', 'flipped', 'kalman_jerk_std', 'kalman_noise', 'framerate', 'time_scale', 'heading_pad']
            dtypes = [float, float, float, float, bool, bool, bool, float, float, float, float, float]
            for key, var, dtype in zip(keys, vars, dtypes):
                if key in config['video_parameters'].keys():
                    if dtype == bool:
                        val = config['video_parameters'][key] == 'True'
                    else:
                        val = dtype(config['video_parameters'][key])
                    setattr(self, var, val)
            # and grab the stored marker parameters
            self.marker_coords = {}
            for key, val in config['fly_markers'].items():
                vals = val.split(",")
                vals = [int(val) for val in vals]
                self.marker_coords[key] = np.array(vals)
        self.display_interval = 1000. / self.display_rate
        # optionally rotate the image
        # if self.rotate270:
        #     h, w = self.img_width, self.img_height
        #     self.img_width = h
        #     self.img_height = w
        self.heading = 0

    def setup(self, linewidth=10):
        # create an empty widget
        self.widget = QtWidgets.QWidget()
        # labels
        # configure
        pg.setConfigOptions(antialias=True)
        # create a graphics layout widget
        self.window = pg.GraphicsLayoutWidget()
        # add a view box
        self.view_objective = self.window.addViewBox(row=0, col=0)
        self.view_relative = self.window.addViewBox(row=0, col=1)
        # self.view_relative = MarkedViewBox(gui=self, row=0, col=1)
        # self.window.addItem(self.view_relative, row=0, col=1, rowspan=1, colspan=1)
        # make the raw heading data a low opacity gray line
        self.heading_plot = self.window.addPlot(row=1, col=0, colspan=2)
        self.heading_plot_raw = self.heading_plot.plot(x=[], y=[], pen=pg.mkPen(.2))
        # and add a full opacity line for the Kalman Filter output
        self.heading_plot_smooth = self.heading_plot.plot(x=[], y=[])
        # self.heading_plot.setOpacity(.25)
        self.heading_plot.setMaximumHeight(200)
        # add ylabel to the left of the left axis saying "unwrapped heading (deg)"
        self.heading_plot.setLabel('left', 'unwrapped heading (°)')
        # todo: make the x-label a button to open a dialog for setting the time scale
        # add a button to set the time scale
        # self.scale_button = QtWidgets.QPushButton("Set Plot Scales")
        # self.scale_button.clicked.connect(self.set_scale)
        # self.heading_plot.layout.addItem(self.scale_button, 3, 0)
        # # add another y-axis on the right for the centered coordinates
        self.heading_relative_axis = pg.AxisItem('right')
        self.heading_plot.layout.addItem(self.heading_relative_axis, 2, 3)
        # self.heading_plot.getAxis('right').linkToView(self.view_relative)
        # set the range of the right axis
        self.heading_relative_axis.setRange(-1.2*self.heading_pad, 1.2*self.heading_pad)
        # add right label
        self.heading_relative_axis.setLabel('centered heading')
        # add a tick for every 45 degrees within the range
        self.heading_relative_axis.setTicks([[(i, str(i)) for i in range(-int(self.heading_pad), int(self.heading_pad)+1, 45)]])
        # self.heading_plot.showAxis('right')
        # place label to the right of the right axis
        # right axis should always go between -self.heading_pad and self.heading_pad
        self.heading_plot.getAxis('right').setRange(-self.heading_pad, self.heading_pad)
        self.heading_plot.getAxis('left').setRange(-self.heading_pad, self.heading_pad)
        # add a label for the shared x- or time axis
        self.heading_plot.setLabel('bottom', 'time (s)')
        # # WBA plots
        # self.wba_plot = self.window.addPlot(row=2, col=0, colspan=2, title='WBA')
        # self.wba_plot_left = self.wba_plot.plot(x=[], y=[], pen=pg.mkPen(255 * blue[0], 255 * blue[1], 255 * blue[2], 150, width=1))
        # # and add a full opacity line for the Kalman Filter output
        # self.wba_plot_right = self.wba_plot.plot(x=[], y=[], pen=pg.mkPen(255 * red[0], 255 * red[1], 255 * red[2], 150, width=1))
        # self.wba_plot.setMaximumHeight(200)
        # WBA plots - L-R and L+R
        if self.wingbeat_analysis:
            self.wba_plot_lmr = self.window.addPlot(row=2, col=0, colspan=2, title='L-R WBA')
            self.wba_lmr = self.wba_plot_lmr.plot(x=[], y=[], pen=pg.mkPen(255 * blue[0], 255 * blue[1], 255 * blue[2], 150, width=1))
            self.wba_plot_lmr.setMaximumHeight(100)
            hori_line_pen = pg.mkPen(0.5, width=1, style=QtCore.Qt.DashLine)
            hori_line = pg.InfiniteLine((0, 0), angle=0, pen=hori_line_pen)
            self.wba_plot_lmr.addItem(hori_line)
            # and add a full opacity line for the Kalman Filter output
            self.wba_plot_lpr = self.window.addPlot(row=3, col=0, colspan=2, title='L+R WBA')
            self.wba_lpr = self.wba_plot_lpr.plot(x=[], y=[], pen=pg.mkPen(255 * red[0], 255 * red[1], 255 * red[2], 150, width=1))
            self.wba_plot_lpr.setMaximumHeight(100)
        # reset the headings data by default
        # add the image item to the viewbox
        if self.rotate270:
            self.image_objective = pg.ImageItem(border='k', axisOrder='row-major')
            self.image_relative = pg.ImageItem(border='k', axisOrder='row-major')
            # self.image_relative = MarkedImageItem(gui=self, border='k', axisOrder='row-major')
        else:
            self.image_objective = pg.ImageItem(border='k')
            self.image_relative = pg.ImageItem(border='k')
        # add image items to the corresponding View boxes
        self.view_objective.addItem(self.image_objective)
        self.view_relative.addItem(self.image_relative)
        max_length = np.sqrt(self.img_width ** 2 + self.img_height ** 2)
        # if self.rotate270:
        # self.center_y, self.center_x = self.img_width / 2, self.img_height / 2
        self.center_y, self.center_x = 0, 0
        # self.view_objective.setLimits(xMin=0, xMax=self.img_height,
        #                               yMin=0, yMax=self.img_width)
        self.view_objective.setRange(xRange=(-max_length/2, max_length/2),
                                        yRange=(-max_length/2, max_length/2))
        self.view_relative.disableAutoRange()
        # self.view_relative.setLimits(xMin=-max_length/2, xMax=max_length/2,
        #                              yMin=-max_length/2, yMax=max_length/2)
        self.view_relative.setRange(xRange=(-max_length/2, max_length/2),
                                    yRange=(-max_length/2, max_length/2))
        # else:
        #     self.center_x, self.center_y = self.img_width/2, self.img_height/2
        #     self.view_objective.setLimits(xMin=0, xMax=self.img_width,
        #                                   yMin=0, yMax=self.img_height)
        # lock the aspect ratio
        self.view_objective.setAspectLocked(True)
        self.view_relative.setAspectLocked(True)
        ## Ring and Arrow Graphics ##
        self.setup_rings_and_arrow(linewidth=linewidth)
        ## Settings Menu ##
        # create a horizontal layout for placing the two panels
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.window, 4)
        # create a vertical layout for placing the different settings menus
        self.menu_widget = QtWidgets.QWidget()
        self.menu_widget.setFixedWidth(250)
        self.menu_layout = QtWidgets.QVBoxLayout()
        self.menu_widget.setLayout(self.menu_layout)
        self.layout.addWidget(self.menu_widget)
        # add layout to the main widget
        self.widget.setLayout(self.layout)
        # add the window and any other widgets to the layout
        self.layout.addLayout(self.menu_layout, 4)
        ## Slider Options ##
        self.setup_sliders()
        ## todo: Exposure Options ##
        if self.wingbeat_analysis:
            # setup the marker information 
            self.setup_fly_markers()
        ## Kalman Options ##
        self.setup_kalman_menu()
        ## Log Options ##
        self.setup_trial_options()
        # set widget as main widget
        self.setCentralWidget(self.widget)
        # set background to black
        self.setStyleSheet("color: white;")
        self.setStyleSheet("background-color: black;")

    def set_scale(self):
        """Open a dialog window to allow custom time and heading scales."""
        # open a new dialog window
        self.scale_dialog = QtWidgets.QDialog()
        self.scale_dialog.setWindowTitle("Set Scale")
        # create a horizontal layout for each slider
        self.scale_layout = QtWidgets.QHBoxLayout()
        # create a slider for the time scale
        self.time_scale_slider = Slider(1, 1000, var_lbl="Time Scale")
        self.time_scale_slider.slider.setValue(self.time_scale)
        self.time_scale_slider.setLabelValue(self.update_time_scale)
        # add tick marks for [10, 20, 60, 120, 240, 480, 960] seconds
        self.time_scale_slider.slider.setTickInterval(10)
        self.time_scale_slider.slider.setTickPosition(QtWidgets.QSlider.TicksLeft)
        # create a slider for the heading scale
        # self.heading_scale_slider = Slider(1, 1000, var_lbl="Heading Scale")


    def setup_rings_and_arrow(self, linewidth=10):
        # inner ring
        self.inner_ring = QtWidgets.QGraphicsEllipseItem(
            self.inner_radius, self.inner_radius,
            self.inner_radius * 2, self.inner_radius * 2)
        self.inner_ring.setPen(
            pg.mkPen(255 * red[0], 255 * red[1], 255 * red[2], 150,
                     width=linewidth))
        self.inner_ring.setBrush(pg.mkBrush(None))
        self.view_objective.addItem(self.inner_ring)
        # outer ring
        self.outer_ring = QtWidgets.QGraphicsEllipseItem(
            self.outer_radius, self.outer_radius,
            self.outer_radius * 2, self.outer_radius * 2)
        self.outer_ring.setPen(
            pg.mkPen(255 * green[0], 255 * green[1], 255 * green[2], 150,
                     width=linewidth))
        self.outer_ring.setBrush(pg.mkBrush(None))
        self.view_objective.addItem(self.outer_ring)
        if self.wingbeat_analysis:
            # wing ring
            self.wing_ring = QtWidgets.QGraphicsEllipseItem(
                self.wing_radius, self.wing_radius,
                self.wing_radius * 2, self.wing_radius * 2)
            self.wing_ring.setPen(
                pg.mkPen(255 * blue[0], 255 * blue[1], 255 * blue[2], 150,
                        width=linewidth))
            self.wing_ring.setBrush(pg.mkBrush(None))
            self.view_objective.addItem(self.wing_ring)
        # plot the heading point and line
        posy = self.inner_radius - linewidth/2
        # plot the heading vector
        # heading line
        self.pt_center = np.array(
            [0, 0])
        self.pt_head = np.array(
            [0, posy])
        self.head_line = pg.PlotCurveItem()
        self.head_pen = pg.mkPen(255 * red[0], 255 * red[1], 255 * red[2], 150,
                                 width=linewidth/2)
        self.standby_pen = pg.mkPen(100, 100, 100, 150,
                                    width=linewidth/2)
        self.head_line.setData(
            x=np.array([self.pt_center[0], self.pt_head[0]]),
            y=np.array([self.pt_center[1], self.pt_head[1]]),
            pen=self.standby_pen)
        self.view_objective.addItem(self.head_line)
        # center circle
        self.center_pin = QtWidgets.QGraphicsEllipseItem(
            -linewidth/2, -linewidth/2, linewidth, linewidth)
        self.center_pin.setPen(
            pg.mkPen(255 * red[0], 255 * red[1], 255 * red[2], 255,
                     width=1))
        self.center_pin.setBrush(
            pg.mkBrush(255 * red[0], 255 * red[1], 255 * red[2], 255))
        self.view_objective.addItem(self.center_pin)
        # heading circle
        self.head_pin = QtWidgets.QGraphicsEllipseItem(
            -linewidth/2, -linewidth/2, linewidth, linewidth)
        self.head_pin.setPen(
            pg.mkPen(255 * red[0], 255 * red[1], 255 * red[2], 255,
                     width=1))
        self.head_pin.setBrush(
            pg.mkBrush(255 * red[0], 255 * red[1], 255 * red[2], 255))
        self.view_objective.addItem(self.head_pin)
        # the center-of-mass point
        self.com_pin = QtWidgets.QGraphicsEllipseItem(
            0, 0, linewidth/2, linewidth/2)
        self.com_pin.setPen(
            pg.mkPen(255 * red[0], 255 * red[1], 255 * red[2], 255,
                     width=1))
        self.com_pin.setBrush(
            pg.mkBrush(255 * red[0], 255 * red[1], 255 * red[2], 255))
        # self.com_pin.setPen(pg.mkPen(255, 255, 255, 255, width=1))
        # self.com_pin.setBrush(
        #     pg.mkBrush(255, 255, 255, 255))
        self.view_objective.addItem(self.com_pin)
        # plot the cardinal directions
        self.plot_cardinal_directions()

    def plot_cardinal_directions(self):
        """This will (re)plot the cardinal directions based on the size of the incoming image."""
        # remove any cardinal labels if present
        if "cardinal_labels" in dir(self):
            for label in self.cardinal_labels:
                self.view_objective.removeItem(label)
            del self.cardinal_labels
        # plot ticks at the extent of the 0, pi/2, pi, and 2*pi/2 axes
        radius = (max(self.img_height, self.img_width))/2 + self.border_pad + 50
        self.cardinal_labels = []
        for ang, lbl in zip([0, np.pi/2, np.pi, 3*np.pi/2], ["0", "90", "180", "270"]):
            x, y = radius*np.cos(ang), radius*np.sin(ang)
            label = pg.TextItem(lbl, color='white', anchor=(.5, .5), angle=(ang - np.pi/2) * 180 / np.pi)
            self.view_objective.addItem(label)
            label.setPos(x, y)
            self.cardinal_labels += [label]

    def setup_sliders(self):
        ## Ring Detector Options ##
        # make a layout for the sliders
        self.sliders_layout = QtWidgets.QHBoxLayout()
        # self.sliders_widget = QtWidgets.QWidget()
        # self.sliders_widget.setLayout(self.sliders_layout)
        # self.menu_layout.addWidget(self.sliders_widget)
        # make three sliders:
        # 1. for controlling the pixel value threshold
        self.threshold_slider = Slider(0, 255, var_lbl='thresh:')
        # self.layout.addWidget(self.threshold_slider, 0, 1, 1, 1)
        self.sliders_layout.addWidget(self.threshold_slider)
        # 2. for controlling the inner ring
        radius_max = min(self.img_height, self.img_width) / 2.
        self.inner_slider = Slider(0, radius_max, var_lbl='head r:')
        # self.layout.addWidget(self.inner_slider, 0, 2, 1, 1)
        self.sliders_layout.addWidget(self.inner_slider)
        # 3. control outer ring
        self.outer_slider = Slider(0, radius_max, var_lbl='tail r:')
        self.sliders_layout.addWidget(self.outer_slider)
        self.menu_layout.addLayout(self.sliders_layout, 1)
        if self.wingbeat_analysis:
            # 4. control wing ring
            self.wing_slider = Slider(0, radius_max, var_lbl='wing r:')
            self.sliders_layout.addWidget(self.wing_slider)
            self.menu_layout.addLayout(self.sliders_layout, 1)
        # connect the radius sliders to the ring radius
        def update_val(value=None, abs_value=None, slider=self.inner_slider,
                       ring=self.inner_ring):
            slider.setLabelValue(value=value, abs_value=abs_value)
            ring.setRect(self.center_x - slider.x, self.center_y - slider.x,
                         2 * slider.x, 2 * slider.x)
            # self.update_plots()
            self.save_vars()
        self.inner_slider.slider.valueChanged.connect(
            partial(update_val, slider=self.inner_slider, ring=self.inner_ring))
        self.outer_slider.slider.valueChanged.connect(
            partial(update_val, slider=self.outer_slider, ring=self.outer_ring))
        if self.wingbeat_analysis:
            self.wing_slider.slider.valueChanged.connect(
                partial(update_val, slider=self.wing_slider, ring=self.wing_ring))
            self.wing_slider.setLabelValue(abs_value=self.wing_radius)
        # set the slider to default values
        self.threshold_slider.setLabelValue(abs_value=self.thresh)
        self.inner_slider.setLabelValue(abs_value=self.inner_radius)
        self.outer_slider.setLabelValue(abs_value=self.outer_radius)

    def setup_kalman_menu(self):
        # make a layout for the Kalman Filter
        self.kalman_layout = QtWidgets.QVBoxLayout()
        self.kalman_widget = QtWidgets.QWidget()
        self.kalman_widget.setLayout(self.kalman_layout)
        self.menu_layout.addWidget(self.kalman_widget)
        # add two input boxes to define the model jerk std and the modelled noise std
        label = QtWidgets.QLabel("Kalman Settings")
        label.setStyleSheet("color: white")
        label.setAlignment(QtCore.Qt.AlignBottom)
        # self.kalman_widget.setFixedHeight(100)
        self.kalman_layout.addWidget(label)
        sublayout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Jerk Std.")
        label.setStyleSheet("color: white")
        sublayout.addWidget(label)
        # self.kalman_layout.addWidget(label)
        self.kalman_jerk_std_box = QtWidgets.QLineEdit(str(self.kalman_jerk_std))
        self.kalman_jerk_std_box.setValidator(QtGui.QDoubleValidator())
        self.kalman_jerk_std_box.setMaximumWidth(50)
        self.kalman_jerk_std_box.returnPressed.connect(self.save_vars)
        self.kalman_jerk_std_box.setStyleSheet("color: white")
        sublayout.addWidget(self.kalman_jerk_std_box)
        # self.kalman_layout.addWidget(self.kalman_jerk_std_box)
        label = QtWidgets.QLabel("Noise Std.")
        label.setStyleSheet("color: white")
        sublayout.addWidget(label)
        # self.kalman_layout.addWidget(label)
        self.kalman_noise_box = QtWidgets.QLineEdit(str(self.kalman_noise))
        self.kalman_noise_box.setValidator(QtGui.QDoubleValidator())
        self.kalman_noise_box.setMaximumWidth(50)
        self.kalman_noise_box.returnPressed.connect(self.save_vars)
        self.kalman_noise_box.setStyleSheet("color: white")
        sublayout.addWidget(self.kalman_noise_box)
        self.kalman_layout.addLayout(sublayout)
        # self.menu_layout.addLayout(self.kalman_layout, 1)

    def setup_trial_options(self):
        config = configparser.ConfigParser()
        config.read(self.config_fn)
        # make a layout for the buttons
        self.buttons_layout = QtWidgets.QVBoxLayout()
        # add layouts to the main layout hierarchicaly
        self.menu_layout.addLayout(self.buttons_layout, 1)
        # self.buttons_layout.addLayout(self.kalman_layout, 8)
        # add checkboxes for options
        # invert image:
        self.invert_check = QtWidgets.QCheckBox("invert")
        self.invert_check.setChecked(self.invert)
        # self.invert_check.setCheckState(self.invert)
        self.invert_check.toggled.connect(self.save_vars)
        self.invert_check.setStyleSheet("color: white")
        self.buttons_layout.addWidget(self.invert_check, 1)
        # threshold image (default to False):
        self.thresholding_check = QtWidgets.QCheckBox("threshold")
        self.thresholding_check.setStyleSheet("color: white")
        self.buttons_layout.addWidget(self.thresholding_check, 1)
        # threshold image (default to False):
        self.flip_check = QtWidgets.QCheckBox("flip")
        self.flip_check.setStyleSheet("color: white")
        self.flip_check.toggled.connect(self.save_vars)
        self.buttons_layout.addWidget(self.flip_check, 1)
        # add an input box for each experiment parameter in the config file
        for param in config["experiment_parameter_options"]:
            combobox = QtWidgets.QComboBox()
            vals = [val for val in config['experiment_parameter_options'][param].split(',')]
            combobox.addItems(vals)
            combobox.setStyleSheet("color: white; background-color: black")
            # set the combobox based on the saved experiment parameter
            current_val = config['experiment_parameters'][param]
            combobox.setCurrentText(current_val)
            combobox.currentIndexChanged.connect(self.save_vars)
            self.experiment_parameters[param] = combobox
            self.buttons_layout.addWidget(combobox, 1)

    def setup_fly_markers(self, size=10, cmap='viridis'):
        """Place markers provided.

        This will also plot the wing edges relative to the wing hinges.
        To do this, PIPE both wing angles as a fload with the self.data[wing_{side}]


        Parameters
        ----------
        size : int
            The size of the marker.
        cmap : str
            The colormap to use for the markers.        
        """
        # todo: add a custom mouseClickRelease method and list of markers
        # to the view_relative window
        # make a layout for the buttons
        self.marker_menu = QtWidgets.QVBoxLayout()
        # add layouts to the main layout hierarchicaly
        self.menu_layout.addLayout(self.marker_menu, 1)
        # get nice colors for the markers using the default colormap
        cmap = matplotlib.colormaps[cmap]
        num_markers = len(self.marker_coords.keys())
        colors = cmap(np.linspace(0, 1, num_markers))
        # plot each of the markers and store as an attribute
        for num, ((key, coords), color) in enumerate(zip(self.marker_coords.items(), colors)):
            # add the marker to both the subjective view and marker menu
            self.markers[key] = Marker(key, coords, color, self.marker_menu, 
                                       self.view_relative, self)
        # plot the wing edges as lines starting at the left and right wing marker coordinates
        for lbl, marker in self.markers.items():
            if 'hinge' in lbl:
                edge_side = lbl.split("_")[-1]
                # using the angle, find the second point
                var = "_".join([edge_side, "wing", "amp"])
                if var in dir(self):
                    edge_angle = self.__getattribute__(var)
                else:
                    edge_angle = 3*np.pi/2
                center = marker.coord
                extent = self.wing_radius * np.array([np.cos(edge_angle), np.sin(edge_angle)])
                # make the edge line
                color = marker.color
                line = pg.PlotCurveItem()
                pen = pg.mkPen(255 * color[0], 255 * color[1], 255 * color[2], 150,
                               width=size/2)
                marker.line_pen = pen
                line.setData(
                    [center[0], extent[0]],[center[1], extent[1]], 
                    pen=self.standby_pen)
                # store for easy access
                self.wing_edges[edge_side] = line
                self.view_relative.addItem(line)

    def update_frame(self, frame):
        # check if the frame size chaged. if so, we've added a border
        old_shape = (self.img_height, self.img_width)
        if old_shape != frame.shape:
            self.border_pad = int(round(abs(self.img_height - frame.shape[0])/2))
            self.plot_cardinal_directions()
        pad = self.border_pad
        # threshold the image
        if self.thresholding_check.isChecked():
            if pad > 0:
                frame[pad:-pad, pad:-pad] = 255 * (frame[pad:-pad, pad:-pad] > self.threshold_slider.x).astype('uint8')
            else:
                frame = 255 * (frame > self.threshold_slider.x).astype('uint8')
        # invert the values in the frame with the fly
        if self.invert_check.isChecked():
            if pad > 0:
                frame[pad:-pad, pad:-pad] = 255 - frame[pad:-pad, pad:-pad]
            else:
                if frame.ndim > 2:
                    frame[..., :3] = 255 - frame[..., :3]
                else:
                    frame = 255 - frame
        # save the image for troubleshooting
        # plt.imsave("test.png", frame.astype('uint8'))
        # update the image
        if self.rotate270:
            self.image_objective.setImage(frame[::-1, ::-1])
            self.image_relative.setImage(frame[::-1, ::-1])
        else:
            self.image_objective.setImage(frame)
            self.image_relative.setImage(frame)
        
    def update_data(self, **data):
        """Update the data used for plotting."""
        any_changed = False
        for key, vals in data.items():
            # passed data include 1) a frame ('img'), 2) raw and smoothed heading data ('heading', 'heading_smooth')
            # 3) the center of mass shift ('com_shift'), and 4) the wing angle data ('wing_left' and 'wing_right')
            # if not isinstance(vals, list):
            #     print(key, vals)
            if len(vals) > 0:
                try:
                    vals = np.concatenate(vals)
                except:
                    print(vals[-5:])
                if len(vals) > 0:
                    if key != 'img':
                        # make special changes to the heading data
                        if key in ['heading', 'heading_smooth']:
                            # vals = [np.pi - val for val in vals]
                            vals = np.pi - vals
                            data[key] = vals
                        val = vals[-1]
                        # store current value as an attribute
                        self.__setattr__(key, val)
                        if key in self.data.keys():
                            # self.data[key] += [vals]
                            if len(self.data[key]) > 0:
                                self.data[key] = np.concatenate([self.data[key], vals], axis=0)
                            else:
                                self.data[key] = vals
                        else:
                            self.data[key] = vals
                        any_changed = True
        # # go through the heading and heading_smooth datasets, removing any NaNs
        for key in ['heading', 'heading_smooth']:
            # replace NaNs in vals_arr with nearest value
            vals_arr = self.data[key]
            if len(vals_arr) > 0:
                nans = np.isnan(vals_arr)
                if np.any(nans):
                    nan_inds = np.where(nans)[0]
                    non_nans = np.where(~nans)[0]
                    # for each nan_ind, replace it with the nearest non-NaN value
                    for nan_ind in nan_inds:
                        # find the previous non-NaN value
                        prev_non_nan = max(np.where(non_nans[:nan_ind])[0])
                        # replace the NaN with the nearest non-NaN value
                        vals_arr[nan_ind] = vals_arr[prev_non_nan]
                    self.data[key] = vals_arr
        #print(self.wing_left, self.heading)
        if any_changed:
            self.current_time = time.time()

    def update_plots(self):
        """Update the plotted data."""
        # check if the config file has been updated every 5 seconds
        if time.time() - self.config_update_time > 5:
            self.config_update_time = time.time()
            self.import_config()
        if np.isnan(self.heading):
            # print(self.data['heading'])
            self.tracking_active = False
            self.head_line.setPen(self.standby_pen)
        else:
            if not self.tracking_active:
                self.tracking_active = True
                self.head_line.setPen(self.head_pen)
            # center the objective image around (0, 0)
            transform = QtGui.QTransform()
            dy, dx = -self.img_height/2 - self.border_pad, -self.img_width/2 - self.border_pad
            # shift the objective image based on the shift in the center of mass
            if self.com_shift is not None:
                # move the center of mass dot
                com_transform = QtGui.QTransform()
                com_transform.translate(self.com_shift[0], self.com_shift[1])
                self.com_pin.setTransform(com_transform)
            transform.translate(dx, dy)
            # transform.translate(-self.img_height/2, -self.img_width/2)
            self.image_objective.setTransform(transform)
            # update position of head pin based on the heading and inner radius
            dy = self.inner_slider.x * np.sin(self.heading)
            dx = self.inner_slider.x * np.cos(self.heading)
            self.head_pin.setPos(dx, dy)
            # update the position of the second point in the heading line
            self.head_line.setData([0, dx], [0, dy])
            # and rotate the relative image by the heading angle
            transform = QtGui.QTransform()
            transform.rotate(-self.heading * 180 / np.pi + 90)
            dx, dy = -self.img_height/2 - self.border_pad, -self.img_width/2 - self.border_pad
            if self.com_shift is not None:
                dx -= self.com_shift[0]
                dy -= self.com_shift[1]
            transform.translate(dx, dy)
            self.image_relative.setTransform(transform)
            for vars, plot in zip([['heading', 'heading_smooth']], [self.heading_plot]):
                for num, var in enumerate(vars):
                    if var in self.data.keys():
                        vals = self.data[var]
                        if len(vals) > 0:
                            # vals_arr = np.concatenate(vals)
                            vals_arr = vals
                            if var in ['heading', 'heading_smooth']:
                                # replace NaNs in vals_arr with nearest value
                                vals_arr = np.unwrap(vals_arr)
                                # replace any NaNs with the 
                            # if var == 'wing_left':
                            #     vals_arr = 3*np.pi/2 - vals_arr
                            vals_arr *= 180 / np.pi
                            # if np.any(abs(np.diff(vals_arr)) > 45):
                            #     print(vals_arr)
                            if vals is not None:
                                # xs = np.linspace(0, len(vals_arr)/self.framerate, len(vals_arr))
                                # self.heading_plot.dataItems[num].setData(x=xs, y=vals_arr)
                                xs = np.linspace(0, self.current_time - self.start_time, len(vals_arr))
                                plot.dataItems[num].setData(x=xs, y=vals_arr)
                            # if var == 'heading' and self.heading_plot.saveState()['view']['autoRange'] == [True, True]:
                            if var == 'heading':
                                # if relative axis is hidden, show it
                                self.heading_relative_axis.show()
                                miny, maxy = vals_arr[-1] - self.heading_pad, vals_arr[-1] + self.heading_pad
                                # and the y ticks to be multiples of 45 degrees within the range
                                # find the lowest multiple of 45 that is greater than miny
                                # update the y-axis range to be +/- self.heading_pad of the last value
                                plot.setYRange(miny, maxy)
                                # update the x-axis range to be the last 10 seconds before the current time
                                minx, maxx = self.current_time - self.start_time - self.time_scale, self.current_time - self.start_time
                                plot.setXRange(minx, maxx)
                                # and the x ticks to be every second
                                time_scale = self.time_scale
                            # else:
                            #     miny, maxy = plot.viewRange()[1]
                            #     # if autorange is on, hide the relative axis
                            #     self.heading_relative_axis.hide()
                            #     # plot x and y ticks based on the auto ranges
                            #     minx, maxx = 0, self.current_time - self.start_time
                            #     time_scale = self.current_time - self.start_time
                            #     plot.setXRange(minx, maxx)
                            # and the x ticks to be relative to the time range
                            if time_scale <= 10:
                                tick_interval = 1
                            elif time_scale <= 60:
                                tick_interval = 5
                            else:
                                tick_interval = 10
                            # set the y ticks to be multiples of 45 degrees within the range
                            miny_rnd = int(45 * np.ceil(miny / 45))
                            maxy_rnd = int(45 * np.floor(maxy / 45))
                            plot.getAxis('left').setTicks([[(i, str(i)) for i in range(miny_rnd, maxy_rnd + 1, 45)]])
                            # similarly, set the x ticks to be multiples of tick_interval within the range
                            minx_rnd = max(0, int(tick_interval * np.ceil(minx / tick_interval)))
                            maxx_rnd = int(tick_interval * np.floor(maxx / tick_interval))
                            plot.getAxis('bottom').setTicks([[(i, str(i)) for i in range(minx_rnd, maxx_rnd+tick_interval, tick_interval)]])
            # calculate and plot the L-R and L+R WBA data
            if 'wing_left' in self.data.keys() and 'wing_right' in self.data.keys() and self.wingbeat_analysis:
                left_wing, right_wing = np.concatenate(self.data['wing_left']), np.concatenate(self.data['wing_right'])
                # print(self.data['wing_left'], self.data['wing_right'])
                # window = 11
                # weights = np.arange(window).astype(float)
                # weights /= weights.sum()
                # left_wing = np.convolve(left_wing, weights, 'valid') * 180 / np.pi
                # right_wing = np.convolve(right_wing, weights, 'valid') * 180 / np.pi
                # left_wing = np.cumsum(left_wing) * 180 / np.pi
                # right_wing = np.cumsum(right_wing) * 180 / np.pi
                lmr = -(left_wing + right_wing)  # flip lmr so that the sign matches the direction of turning
                lpr = 360 - (abs(left_wing) + abs(right_wing))
                xs = np.linspace(0, self.current_time - self.start_time, len(lmr))
                self.wba_plot_lmr.dataItems[0].setData(x=xs, y=lmr)
                self.wba_plot_lpr.dataItems[0].setData(x=xs, y=lpr)
        if 'left' in self.wing_edges.keys() and 'right' in self.wing_edges.keys() and self.wingbeat_analysis:
            for side in ['left', 'right']:
                angle = 0
                if f"wing_{side}" in self.data.keys():
                    angles = self.data[f"wing_{side}"][-1]
                    no_nans = np.isnan(angles) == False
                    if np.any(no_nans):
                        angle = angles[np.argmax(abs(angles[no_nans]))]
                angle += 3*np.pi / 2
                # invert for the left plot
                edge = self.wing_edges[side]
                marker = self.markers[f"wing_hinge_{side}"]
                    # change the pen to standby if the angle is nan
                if np.isnan(angle):
                    edge.setPen(self.standby_pen)
                    # use the last angle
                else:
                    edge.setPen(marker.line_pen)
                    # update the edge angle
                    extent = self.wing_radius * np.array([np.cos(angle), np.sin(angle)])
                    edge.setData(
                        [marker.coord[0], extent[0]], [marker.coord[1], extent[1]])

    def save_vars(self):
        config = configparser.ConfigParser()
        config.read(self.config_fn)
        vars = ['inner_r', 'outer_r', 'thresh'] 
        vals = [self.inner_slider.x, self.outer_slider.x, self.threshold_slider.x]
        if self.wingbeat_analysis:
            vars += ['wing_r']
            vals += [self.wing_slider.x]
        for var, val in zip(vars, vals):
            config.set('video_parameters', var, str(val))
        # set the new values
        if 'invert_check' in dir(self):
            config.set('video_parameters', 'invert',
                    str(self.invert_check.isChecked()))
        if 'flip_check' in dir(self):
            config.set('video_parameters', 'flipped',
                    str(self.flip_check.isChecked()))
        # save experiment parameter options
        for key, val in self.experiment_parameters.items():
            config.set('experiment_parameters', key, val.currentText())
        # set the kalman settings
        for var, val in zip(['kalman_jerk_std', 'kalman_noise'], ['kalman_jerk_std_box', 'kalman_noise_box']):
            if val in dir(self):
                new_val = self.__getattribute__(val).text()
                if float(new_val) > 0 and float(new_val) != np.nan:
                    config.set('video_parameters', var, new_val)
        # update marker_coords using the stored markers
        for lbl, marker in self.markers.items():
            self.marker_coords[lbl] = marker.coord
        # save the marker coordinates
        for key, coord in self.marker_coords.items():
            # convert to a string
            string = f"{int(coord[0])},{int(coord[1])}"
            # replace the old coordinate
            config.set('fly_markers', key, string)
        # save config file
        with open(self.config_fn, 'w') as configfile:
            config.write(configfile)
    
    def reset_plots(self):
        """Reset the line plot data."""
        # reset the plots
        self.start_time = time.time()
        self.current_time = time.time()
        for key in self.data.keys():
            # empty each array other than the image
            if key != 'img':
                self.data[key] = []
        


class FrameUpdater():
    def __init__(self, height, width, display_rate=60., buffer_fn='_buffer.npy', config_fn='./video_player.config', wingbeat_analysis=False):
        self.buffer_fn = buffer_fn
        self.height = height
        self.width = width
        self.interval = int(round(1000./display_rate))
        self.ind = 0
        self.gui = VideoGUI(img_height=self.height, img_width=self.width, config_fn=config_fn, wingbeat_analysis=wingbeat_analysis)
        # setup a QTimer to update the frame values instead of a simple for loop
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.interval)
        if isinstance(buffer_fn, str):
            self.timer.timeout.connect(self.update_val_from_file)
        else:
            self.timer.timeout.connect(self.update_val_from_PIPE)
        self.timer.start()

    def __del__(self):
        if 'timer' in dir(self):
            if self.timer.isActive():
                self.timer.stop()

    def update_val_from_file(self):
        try:
            # grab packets of data from the saved ndarray
            val = np.load(self.buffer_fn)
            # update the frame
            self.gui.update_frame(val)
            # update the heading variable
            heading = np.load("_heading.npy")
            # get the test headings data
            fn = "_headings.npy"
            fn_smooth = "_headings_smooth.npy"
            if os.path.exists(fn):
                headings = np.load(fn)
            if os.path.exists(fn_smooth):
                headings_smooth = np.load(fn_smooth)
            self.gui.update_plots(headings, headings_smooth)
        except:
            pass

    def update_val_from_PIPE(self):
        buffer_size = sys.stdin.buffer.read(4)
        while not buffer_size:
            buffer_size = sys.stdin.buffer.read(4)
        length_prefix = struct.unpack('I', buffer_size)[0]
        # grab the data from the buffer
        buffer = sys.stdin.buffer.read(length_prefix)
        data = pickle.loads(buffer)
        if 'com_shift' not in data.keys():
            data['com_shift'] = [0]
        # if data is a reset signal, delete all stored data
        if 'reset' in data.keys():
            self.gui.reset_plots()
        else:
            # otherwise, if data is the supplied data, then update the plots
            # extract the new frame and the headings data
            img = data['img']
            if isinstance(img, cp.ndarray) and cupy_loaded:
                img = cp.asnumpy(img)
            # heading, heading_smooth, com_shift = data['heading'].get(), data['heading_smooth'].get(), data['com_shift']
            # split the above into 3 lines
            heading = data['heading']
            heading_smooth = data['heading_smooth']
            com_shift = data['com_shift']
            wing_left, wing_right = data['wing_left'], data['wing_right']
            # print(f"shift = {com_shift}")
            # update the frame and heading
            self.gui.update_frame(img)
            # update the stored data
            self.gui.update_data(heading=heading, heading_smooth=heading_smooth, com_shift=com_shift, wing_left=wing_left, wing_right=wing_right)
            self.gui.update_plots()


if __name__ == "__main__":
    # get optional arguments
    arg_list = list(sys.argv[1:])
    # check for the height and width arguments
    arg_dict = {}
    arg_dict['height'], arg_dict['width'] = 480, 640
    for key, var, dtype in zip(
            ['-h', '-w', '-config_fn', '-wingbeat'], ['height', 'width', 'config_fn', 'wingbeat_analysis'], [int, int, str, bool]):
        inds = [key == arg for arg in arg_list]
        if any(inds):
            ind = np.where(inds)[0][0]
            if dtype == bool:
                arg_dict[var] = dtype(arg_list[ind + 1] == 'True')
            else:
                arg_dict[var] = dtype(arg_list[ind + 1])
    app = pg.mkQApp()
    animation = FrameUpdater(buffer_fn=None, **arg_dict)
    sys.exit(app.exec_())
