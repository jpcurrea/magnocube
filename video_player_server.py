from functools import partial
import configparser
import numpy as np
import os
import pickle
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import sys
from matplotlib import pyplot as plt

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
                 thresholding=True, config_fn="video_player.config"):
        self.config_fn = config_fn
        # grab parameters
        self.img_height, self.img_width = img_height, img_width
        self.thresh = thresh
        self.thresholding = thresholding
        self.inner_radius, self.outer_radius = inner_radius, outer_radius
        self.tracking_active = False
        # read/make config file
        if os.path.exists(config_fn):
            # load using config parser
            config = configparser.ConfigParser()
            config.read(config_fn)
            # check if parameters stored
            keys = ['inner_r', 'outer_r', 'thresh', 'invert', 'rotate270', 'flipped']
            vars = ['inner_radius', 'outer_radius', 'thresh', 'invert',
                    'rotate270', 'flipped']
            dtypes = [float, float, float, bool, bool, bool]
            for key, var, dtype in zip(keys, vars, dtypes):
                if key in config['video_parameters'].keys():
                    if dtype == bool:
                        val = config['video_parameters'][key] == 'True'
                    else:
                        val = dtype(config['video_parameters'][key])
                    setattr(self, var, val)
        self.display_interval = 1000. / display_rate
        # optionally rotate the image
        # if self.rotate270:
        #     h, w = self.img_width, self.img_height
        #     self.img_width = h
        #     self.img_height = w
        self.heading = 0
        self.north = 0
        super().__init__()
        # set the title
        self.setWindowTitle("Live Feed")
        # set the geometry
        # self.setGeometry(100, 100, img_width, img_height)
        # setup the components
        self.setup()
        # show all the widgets
        self.show()

    def setup(self, linewidth=10):
        # create an empty widget
        self.widget = QtWidgets.QWidget()
        # labels
        # configure
        pg.setConfigOptions(antialias=False)
        # create a graphics layout widget
        self.window = pg.GraphicsLayoutWidget()
        # add a view box
        self.view = self.window.addViewBox()
        # add the image item to the viewbox
        if self.rotate270:
            self.image = pg.ImageItem(border='k', axisOrder='row-major')
        else:
            self.image = pg.ImageItem(border='k')
        self.view.addItem(self.image)
        if self.rotate270:
            self.view.setLimits(xMin=0, xMax=self.img_height,
                                yMin=0, yMax=self.img_width)
        else:
            self.view.setLimits(xMin=0, xMax=self.img_width,
                                yMin=0, yMax=self.img_height)
        # lock the aspect ratio
        self.view.setAspectLocked(True)
        # plot the 2 circles
        if self.rotate270:
            self.center_y, self.center_x = self.img_width / 2, self.img_height / 2
        else:
            self.center_x, self.center_y = self.img_width/2, self.img_height/2
        # inner ring
        self.inner_ring = pg.QtGui.QGraphicsEllipseItem(
            self.center_x - self.inner_radius, self.center_y - self.inner_radius,
            self.inner_radius * 2, self.inner_radius * 2)
        self.inner_ring.setPen(
            pg.mkPen(255 * blue[0], 255 * blue[1], 255 * blue[2], 150,
                     width=linewidth))
        self.inner_ring.setBrush(pg.mkBrush(None))
        self.view.addItem(self.inner_ring)
        # outer ring
        self.outer_ring = pg.QtGui.QGraphicsEllipseItem(
            self.center_x - self.outer_radius, self.center_y - self.outer_radius,
            self.outer_radius * 2, self.outer_radius * 2)
        self.outer_ring.setPen(
            pg.mkPen(255 * green[0], 255 * green[1], 255 * green[2], 150,
                     width=linewidth))
        self.outer_ring.setBrush(pg.mkBrush(None))
        self.view.addItem(self.outer_ring)
        # plot the heading point and line
        posy = self.center_y + self.inner_radius - linewidth/2
        # plot the heading vector
        # heading line
        self.pt_center = np.array(
            [self.center_x, self.center_y])
        self.pt_head = np.array(
            [self.center_x, posy])
        self.head_line = pg.PlotCurveItem()
        self.head_pen = pg.mkPen(255 * red[0], 255 * red[1], 255 * red[2], 150,
                                 width=linewidth/2)
        self.standby_pen = pg.mkPen(100, 100, 100, 150,
                                    width=linewidth/2)
        self.head_line.setData(
            x=np.array([self.pt_center[0], self.pt_head[0]]),
            y=np.array([self.pt_center[1], self.pt_head[1]]),
            pen=self.standby_pen)
        self.view.addItem(self.head_line)
        # center circle
        self.center_pin = pg.QtGui.QGraphicsEllipseItem(
            self.center_x - linewidth/2, self.center_y - linewidth/2,
            linewidth, linewidth)
        self.center_pin.setPen(
            pg.mkPen(255 * red[0], 255 * red[1], 255 * red[2], 255,
                     width=1))
        self.center_pin.setBrush(
            pg.mkBrush(255 * red[0], 255 * red[1], 255 * red[2], 255))
        self.view.addItem(self.center_pin)
        # heading circle
        self.head_pin = pg.QtGui.QGraphicsEllipseItem(
            self.center_x - linewidth/2, self.center_y - linewidth/2,
            linewidth, linewidth)
        self.head_pin.setPen(
            pg.mkPen(255 * red[0], 255 * red[1], 255 * red[2], 255,
                     width=1))
        self.head_pin.setBrush(
            pg.mkBrush(255 * red[0], 255 * red[1], 255 * red[2], 255))
        self.view.addItem(self.head_pin)
        # virtual orientation
        self.north_pin = pg.QtGui.QGraphicsEllipseItem(
            self.center_x - linewidth/4, self.center_y - linewidth/4,
            linewidth/2, linewidth/2)
        self.north_pin.setPen(
            pg.mkPen(255, 255, 255, 255, width=1))
        self.north_pin.setBrush(
            pg.mkBrush(255, 255, 255, 255))
        self.view.addItem(self.north_pin)
        # create a horizontal layout
        self.layout = QtWidgets.QHBoxLayout()
        # add layout to the main widget
        self.widget.setLayout(self.layout)
        # add the window and any other widgets to the layout
        # self.layout.addWidget(self.window, 0, 0, 12, 1)
        self.layout.addWidget(self.window, 4)
        # make a layout for the buttons
        self.buttons_layout = QtWidgets.QVBoxLayout()
        # make a layout for the sliders
        self.sliders_layout = QtWidgets.QHBoxLayout()
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
        # add layouts to the main layout hierarchicaly
        self.layout.addLayout(self.buttons_layout, 1)
        self.buttons_layout.addLayout(self.sliders_layout, 8)
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
        self.flip_check.setChecked(self.flipped)
        self.buttons_layout.addWidget(self.flip_check, 1)
        # add an input box
        self.sex_selector = QtWidgets.QComboBox()
        self.sex_selector.addItems(['female', 'male'])
        self.sex_selector.setStyleSheet("color: white")
        self.sex = self.sex_selector.currentText()
        self.buttons_layout.addWidget(self.sex_selector, 1)

        # connect the radius sliders to the ring radius
        def update_val(value=None, abs_value=None, slider=self.inner_slider,
                       ring=self.inner_ring):
            slider.setLabelValue(value=value, abs_value=abs_value)
            ring.setRect(self.center_x - slider.x, self.center_y - slider.x,
                         2 * slider.x, 2 * slider.x)
            self.update_heading()
            self.update_north()
            self.save_vars()
        self.inner_slider.slider.valueChanged.connect(
            partial(update_val, slider=self.inner_slider, ring=self.inner_ring))
        self.outer_slider.slider.valueChanged.connect(
            partial(update_val, slider=self.outer_slider, ring=self.outer_ring))
        # set widget as main widget
        self.setCentralWidget(self.widget)
        # set the slider to default values
        self.threshold_slider.setLabelValue(abs_value=self.thresh)
        self.inner_slider.setLabelValue(abs_value=self.inner_radius)
        self.outer_slider.setLabelValue(abs_value=self.outer_radius)
        # set background to black
        self.setStyleSheet("color: white;")
        self.setStyleSheet("background-color: black;")

    def update_frame(self, frame):
        if self.thresholding_check.isChecked():
            frame = 255 * (frame > self.threshold_slider.x).astype('uint8')
        if self.invert_check.isChecked():
            frame = 255 - frame
        # todo: what's going on with the saved frames? It looks like they're
        # being distorted but the final video is perfect
        # plt.imshow(frame)
        # plt.show()
        if self.rotate270:
            self.image.setImage(frame[::-1, ::-1])
        else:
            self.image.setImage(frame)

    def update_heading(self, heading=None):
        """Update the plotted heading point and line."""
        if heading is None:
            heading = self.heading
        # flip to match the camera and projector coordinates
        # heading *= -1
        heading = np.pi - heading
        # if self.rotate270:
        #     heading += np.pi/2
        if np.isnan(heading):
            self.tracking_active = False
            self.head_line.setPen(self.standby_pen)
        else:
            if not self.tracking_active:
                self.tracking_active = True
                self.head_line.setPen(self.head_pen)
            # update position of head pin based on the heading and inner radius
            dy = self.inner_slider.x * np.sin(heading)
            dx = self.inner_slider.x * np.cos(heading)
            self.head_pin.setPos(dx, dy)
            # update the position of the second point in the heading line
            new_y, new_x = self.center_y + dy, self.center_x + dx
            self.head_line.setData([self.center_x, new_x], [self.center_y, new_y])

    def update_north(self, north=None):
        """Update the plotted north point and line."""
        if north is None:
            north = self.north
        north *= -1
        if self.rotate270:
            north += np.pi/2
        # update position of north pin
        dy = self.inner_slider.x * np.sin(north)
        dx = self.inner_slider.x * np.cos(north)
        self.north_pin.setPos(dx, dy)
        # update the position of the second point in the north line
        new_y, new_x = self.center_y + dy, self.center_x + dx

    def save_vars(self):
        config = configparser.ConfigParser()
        config.read(self.config_fn)
        # set the new values
        config.set('video_parameters', 'inner_r', str(self.inner_slider.x))
        config.set('video_parameters', 'outer_r', str(self.outer_slider.x))
        config.set('video_parameters', 'thresh', str(self.threshold_slider.x))
        config.set('video_parameters', 'invert',
                   str(self.invert_check.isChecked()))
        config.set('experiment_parameters', 'sex', self.sex)
        # save config file
        with open(self.config_fn, 'w') as configfile:
            config.write(configfile)

class FrameUpdater():
    def __init__(self, height, width, display_rate=30., buffer_fn='_buffer.npy'):
        self.buffer_fn = buffer_fn
        self.height = height
        self.width = width
        self.interval = int(round(1000./display_rate))
        self.ind = 0
        self.gui = VideoGUI(
            img_height=self.height, img_width=self.width)
        # setup a QTimer to update the frame values instead of a simple for loop
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.interval)
        self.timer.timeout.connect(self.update_val)
        self.timer.start()

    def __del__(self):
        if self.timer.isActive():
            self.timer.stop()

    def update_val(self):
        try:
            # grab packets of data from the saved ndarray
            val = np.load(self.buffer_fn)
            # update the frame
            self.gui.update_frame(val)
            # update the heading variable
            # heading, north = np.load("_heading.npy")
            heading = np.load("_heading.npy")
            self.gui.update_heading(heading)
            # north = np.load("_north.npy")
            # self.gui.update_north(north)
        except:
            pass

if __name__ == "__main__":
    DATA_FN = "_buffer.npy"
    # get optional arguments
    arg_list = np.array(sys.argv[1:])
    # print(arg_list)
    # check for the height and width arguments
    arg_dict = {}
    arg_dict['height'], arg_dict['width'] = 480, 640
    for key, var in zip(
            ['-h', '-w'], ['height', 'width']):
        if key in list(arg_list):
            ind = np.where(arg_list == key)[0]
            arg_dict[var] = int(arg_list[ind + 1])
    # setup the plotting loop
    app = pg.mkQApp()
    animation = FrameUpdater(
        buffer_fn=DATA_FN, height=int(arg_dict['height']),
        width=int(arg_dict['width']))
    sys.exit(app.exec_())