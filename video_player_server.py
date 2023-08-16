from functools import partial
import configparser
import numpy as np
import os
import pickle
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import threading
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
        self.label.setText(f"{self.var_lbl}\n{self.x:d}")


class VideoGUI(QtWidgets.QMainWindow):
    def __init__(self, img_height=600, img_width=500, display_rate=30,
                 inner_radius=100., outer_radius=150., thresh=100,
                 thresholding=True, config_fn="video_player.config", border_pad=0):
        self.display_rate = display_rate
        self.config_fn = config_fn
        self.experiment_parameters = {}
        self.data = {}
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
        super().__init__()
        # set the title
        self.setWindowTitle("Live Feed")
        # set the geometry
        # self.setGeometry(100, 100, img_width, img_height)
        # setup the components
        self.setup()
        # show all the widgets
        self.show()

    def import_config(self):
        """Import important parameters from the config file."""
        # read/make config file
        if os.path.exists(self.config_fn):
            # load using config parser
            config = configparser.ConfigParser()
            config.read(self.config_fn)
            # self.genotype_list = config.getList('experiment_parameters','genotype')
            # check if parameters stored
            keys = ['inner_r', 'outer_r', 'thresh', 'invert', 'rotate270', 'flipped', 'kalman_jerk_std', 'kalman_noise', 'camera_fps']
            vars = ['inner_radius', 'outer_radius', 'thresh', 'invert', 'rotate270', 'flipped', 'kalman_jerk_std', 'kalman_noise', 'framerate']
            dtypes = [float, float, float, bool, bool, bool, float, float]
            for key, var, dtype in zip(keys, vars, dtypes):
                if key in config['video_parameters'].keys():
                    if dtype == bool:
                        val = config['video_parameters'][key] == 'True'
                    else:
                        val = dtype(config['video_parameters'][key])
                    setattr(self, var, val)
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
        # make the raw heading data a low opacity gray line
        self.heading_plot = self.window.addPlot(row=1, col=0, colspan=2, title='heading')
        self.heading_plot_raw = self.heading_plot.plot(x=[0], y=[0], pen=pg.mkPen(.2))
        # and add a full opacity line for the Kalman Filter output
        self.heading_plot_smooth = self.heading_plot.plot(x=[0], y=[0])
        # self.heading_plot.setOpacity(.25)
        self.heading_plot.setMaximumHeight(200)
        # reset the headings data by default
        # add the image item to the viewbox
        if self.rotate270:
            self.image_objective = pg.ImageItem(border='k', axisOrder='row-major')
            self.image_relative = pg.ImageItem(border='k', axisOrder='row-major')
        else:
            self.image_objective = pg.ImageItem(border='k')
            self.image_relative = pg.ImageItem(border='k')
        # add image items to the corresponding View boxes
        self.view_objective.addItem(self.image_objective)
        self.view_relative.addItem(self.image_relative)
        max_length = np.sqrt(self.img_width ** 2 + self.img_height ** 2)
        if self.rotate270:
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
        else:
            self.center_x, self.center_y = self.img_width/2, self.img_height/2
            self.view_objective.setLimits(xMin=0, xMax=self.img_width,
                                          yMin=0, yMax=self.img_height)
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
        self.menu_widget.setFixedWidth(230)
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
        ## Kalman Options ##
        self.setup_kalman_menu()
        ## Log Options ##
        self.setup_trial_options()
        # set widget as main widget
        self.setCentralWidget(self.widget)
        # set background to black
        self.setStyleSheet("color: white;")
        self.setStyleSheet("background-color: black;")

    def setup_rings_and_arrow(self, linewidth=10):
        # inner ring
        self.inner_ring = QtWidgets.QGraphicsEllipseItem(
            self.inner_radius, self.inner_radius,
            self.inner_radius * 2, self.inner_radius * 2)
        self.inner_ring.setPen(
            pg.mkPen(255 * blue[0], 255 * blue[1], 255 * blue[2], 150,
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
        radius_min = min(self.img_height, self.img_width) / 2.
        self.inner_slider = Slider(0, radius_min, var_lbl='inner r:')
        # self.layout.addWidget(self.inner_slider, 0, 2, 1, 1)
        self.sliders_layout.addWidget(self.inner_slider)
        # 3. control outer ring
        self.outer_slider = Slider(0, radius_min, var_lbl='outer r:')
        self.sliders_layout.addWidget(self.outer_slider)
        self.menu_layout.addLayout(self.sliders_layout, 1)
        # connect the radius sliders to the ring radius
        def update_val(value=None, abs_value=None, slider=self.inner_slider,
                       ring=self.inner_ring):
            slider.setLabelValue(value=value, abs_value=abs_value)
            ring.setRect(self.center_x - slider.x, self.center_y - slider.x,
                         2 * slider.x, 2 * slider.x)
            self.update_plots()
            self.save_vars()
        self.inner_slider.slider.valueChanged.connect(
            partial(update_val, slider=self.inner_slider, ring=self.inner_ring))
        self.outer_slider.slider.valueChanged.connect(
            partial(update_val, slider=self.outer_slider, ring=self.outer_ring))
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
            combobox.currentIndexChanged.connect(self.save_vars)
            self.experiment_parameters[param] = combobox
            self.buttons_layout.addWidget(combobox, 1)
        # add a sex input box
        # self.sex_selector = QtWidgets.QComboBox()
        # self.sex_selector.addItems(['female', 'male'])
        # self.sex_selector.setStyleSheet("color: white; background-color: black")
        # self.sex = self.sex_selector.currentText()
        # self.buttons_layout.addWidget(self.sex_selector, 1)
        # add a genotype input box
        # self.genotype_selector = QtWidgets.QComboBox()
        # genotype_list = json.loads(config.get("experiment_parameters","genotype"))
        # self.genotype_selector.addItems(genotype_list)
        # self.genotype_selector.setStyleSheet("color: white; background-color: black")
        # self.genotype = self.genotype_selector.currentText()
        # self.buttons_layout.addWidget(self.genotype_selector, 1)

    def update_frame(self, frame):
        # check if the frame size chaged. if so, we've added a border
        old_shape = (self.img_height, self.img_width)
        if old_shape != frame.shape:
            self.border_pad = int(round(abs(self.img_height - frame.shape[0])/2))
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
                frame[..., :3] = 255 - frame[..., :3]
        # save the image for troubleshooting
        # plt.imsave("test.png", frame)
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
            if not isinstance(vals, list):
                print(key, vals)
            if len(vals) > 0:
                vals = np.concatenate(vals)
                if len(vals) > 0:
                    if key != 'img':
                        # make special changes to the heading data
                        if key in ['heading', 'heading_smooth']:
                            vals = [np.pi - val for val in vals]
                            data[key] = vals
                        val = vals[-1]
                        # store as an attribute
                        self.__setattr__(key, val)
                        if key in self.data.keys():
                            self.data[key] += [vals]
                        else:
                            self.data[key] = [vals]
                        any_changed = True
        if any_changed:
            self.current_time = time.time()

    def update_plots(self):
        """Update the plotted data."""
        # if heading_smooth is None:
        #     heading_smooth = heading
        # # use smooth headings unless they are nans
        # use_smooth_headings = False
        # if len(headings_smooth) > 0:
        #     if headings_smooth[-1] is not np.nan:
        #         use_smooth_headings = True
        # # add to the stored data
        # if use_smooth_headings:
        #     heading = heading_smooth
        # elif len(headings) > 0:
        #     heading = headings[-1]
        # else:
        #     heading = 0
        # if np.isnan(heading):
        #     heading = 0
        # # combine with the stored data
        # self.headings += [heading]

        # # flip to match the camera and projector coordinates
        # # heading *= -1
        # heading = np.pi - heading
        # headings = np.pi - headings
        # headings_smooth = np.pi - headings_smooth
        # self.heading = heading
        # if self.rotate270:
        #     heading += np.pi/2
        if np.isnan(self.heading):
            print(self.data['heading'])
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
            # plot the data
            # todo: store all of the data to plot (self.data)
            for num, (key, vals) in enumerate(self.data.items()):
                if len(vals) > 0 and key in ['heading', 'heading_smooth']:
                    vals_arr = np.concatenate(vals)
                    if key in ['heading', 'heading_smooth']:
                        diffs = vals_arr[1:] - vals_arr[:-1]
                        diffs[np.isnan(diffs)] = 0
                        new_vals = np.cumsum(diffs)
                        vals_arr = np.unwrap(new_vals)
                    vals_arr *= 180 / np.pi
                    if vals is not None:
                        # xs = np.linspace(0, len(vals_arr)/self.framerate, len(vals_arr))
                        # self.heading_plot.dataItems[num].setData(x=xs, y=vals_arr)
                        xs = np.linspace(0, self.current_time - self.start_time, len(vals_arr))
                        self.heading_plot.dataItems[num].setData(x=xs, y=vals_arr)

    def save_vars(self):
        config = configparser.ConfigParser()
        config.read(self.config_fn)
        vars = ['inner_r', 'outer_r', 'thresh'] 
        vals = [self.inner_slider.x, self.outer_slider.x, self.threshold_slider.x]
        for var, val in zip(vars, vals):
            config.set('video_parameters', var, str(val))
        # set the new values
        if 'invert_check' in dir(self):
            config.set('video_parameters', 'invert',
                    str(self.invert_check.isChecked()))
        # save experiment parameter options
        for key, val in self.experiment_parameters.items():
            config.set('experiment_parameter_options', key, val.currentText())
        # if 'sex' in dir(self):
        #     config.set('experiment_parameters', 'sex', self.sex)
        # if 'genotype' in dir(self):
        #     config.set('experiment_parameters', 'genotype', self.genotype)
        # set the kalman settings
        for var, val in zip(['kalman_jerk_std', 'kalman_noise'], ['kalman_jerk_std_box', 'kalman_noise_box']):
            if val in dir(self):
                new_val = self.__getattribute__(val).text()
                if float(new_val) > 0 and float(new_val) != np.nan:
                    config.set('video_parameters', var, new_val)
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
    def __init__(self, height, width, display_rate=30., buffer_fn='_buffer.npy', config_fn='./video_player.config'):
        self.buffer_fn = buffer_fn
        self.height = height
        self.width = width
        self.interval = int(round(1000./display_rate))
        self.ind = 0
        self.gui = VideoGUI(img_height=self.height, img_width=self.width, config_fn=config_fn)
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
            heading, heading_smooth, com_shift = data['heading'], data['heading_smooth'], data['com_shift']
            # update the frame and heading
            self.gui.update_frame(img)
            # update the stored data
            self.gui.update_data(heading=heading, heading_smooth=heading_smooth, com_shift=com_shift)
            self.gui.update_plots()


if __name__ == "__main__":
    DATA_FN = "_buffer.npy"
    # get optional arguments
    arg_list = list(sys.argv[1:])
    # check for the height and width arguments
    arg_dict = {}
    arg_dict['height'], arg_dict['width'] = 480, 640
    for key, var, dtype in zip(
            ['-h', '-w', '-config_fn'], ['height', 'width', 'config_fn'], [int, int, str]):
        inds = [key in arg for arg in arg_list]
        if any(inds):
            ind = np.where(inds)[0][0]
            arg_dict[var] = dtype(arg_list[ind + 1])
    # start a thread to play an example video 
    # play_video("./revolving_fbar_different_starts/2023_04_27_15_29_05")
    # setup the plotting loop
    app = pg.mkQApp()
    # animation = FrameUpdater(
    #     buffer_fn=DATA_FN, height=int(arg_dict['height']),
    #     width=int(arg_dict['width']))
    animation = FrameUpdater(buffer_fn=None, **arg_dict)
    sys.exit(app.exec_())
