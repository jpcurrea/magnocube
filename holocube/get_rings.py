"""Use a version of tracker_window to select the center and 2 radii for magno tracking.


Each video needs a specific 2D center point and two radii. These are stored in numpy 
array file (e.g. data.npy). If a video isn't yet in the numpy array, add an empty entry.

"""
from scipy import spatial
from collections import namedtuple, Counter
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.patches import Arrow, Circle, Polygon, Rectangle
from matplotlib.animation import FuncAnimation

# from MinimumBoundingBox import MinimumBoundingBox
import matplotlib
from matplotlib import pyplot as plt
import math
import time
import numpy as np
import os
import PIL
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from scipy import interpolate, ndimage, spatial, optimize
import scipy
from skvideo import io
from skimage.feature import peak_local_max
from sklearn import cluster
import subprocess
import threading
import sys

# from points_GUI import *

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

blue, green, yellow, orange, red, purple = [
    (0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (0.83, 0.74, 0.37),
    (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]

colors = [
    'tab:red',
    'tab:green',
    'tab:blue',
    'tab:orange',
    'tab:purple',
    'tab:cyan',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive'
]


def print_progress(part, whole):
    prop = float(part)/float(whole)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), 100*prop))
    sys.stdout.flush()


filetypes = [
    ('matlab videos', '*.mat'),
    ('mpeg videos', '*.mpg *.mpeg *.mp4'),
    ('avi videos', '*.avi'),
    ('quicktime videos', '*.mov *.qt'),
    ('h264 videos', '*.h264'),
    ('all files', '*')
]

# format the filetypes for the pyqt file dialog box
ftypes = []
for (fname, ftype) in filetypes:
    ftypes += [f"{fname} ({ftype})"]
ftypes = ";;".join(ftypes)


class FileSelector(QWidget):
    """Offers a file selection dialog box with filters based on common image filetypes.
    """

    def __init__(self, filetypes=ftypes):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        super().__init__()
        self.title = 'Select the videos you want to process.'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.filetypes = filetypes
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.openFileNamesDialog()
        self.show()

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        self.files, self.ftype = QFileDialog.getOpenFileNames(
            self,
            "QFileDialog.getOpenFileNames()",
            "",
            self.filetypes,
            options=options)


class VideoTrackerWindow():
    def __init__(self, filename, tracking_folder="tracking_data",
                 data_fn_suffix='_track_data.npy', fps=30,
                 small_radius=100, large_radius=200):
        # m.pyplot.ion()
        self.filename = filename
        self.dirname = os.path.dirname(filename)
        self.basename = os.path.basename(filename)
        self.tracking_folder = os.path.join(
            self.dirname,
            tracking_folder)
        self.ftype = self.filename.split(".")[-1]
        self.tracking_fn = os.path.join(
            self.tracking_folder,
            self.basename.replace(f".{self.ftype}", data_fn_suffix))
        self.load_file()
        self.num_frames = len(self.video)
        self.range_frames = np.array(range(self.num_frames))
        self.curr_frame_index = 0
        self.inherent_fps = fps
        # markers and data file
        self.curr_radius = 0
        if os.path.isfile(self.tracking_fn):
            self.data = np.load(self.tracking_fn)
        else:
            self.data = np.zeros(1, dtype=[
                ('x', '<f8'), ('y', '<f8'), ('radius_small', '<f8'), ('radius_large', '<f8')])
            # use center of mass of intersecting region
            foreground = self.video > 50
            foreground = foreground.mean(0)
            com = ndimage.center_of_mass(foreground)
            self.data['x'], self.data['y'] = com[1], com[0]
            # use default radii 
            self.data['radius_small'], self.data['radius_large'] = small_radius, large_radius
        self.small_radius = self.data['radius_small']
        self.large_radius = self.data['radius_large']
        self.center = self.data[['x', 'y']]
        # markers has the shape 2 x 2
        # where the second dimension specifies if the values are a center point
        # or radii. for centers, the 3rd dimension specifies x or y. for the radii,
        # the second dimension specifies the inside or outside ring
        self.data_changed = False
        # the figure
        self.load_image()
        # figsize = self.image.shape[1]/90, self.image.shape[0]/90
        h, w = self.image.shape[:2]
        if w > h:
            fig_width = 10
            fig_height = h/w * fig_width
        else:
            fig_height = 8
            fig_width = w/h * fig_height
        self.figure = plt.figure(1, figsize=(fig_width, fig_height), dpi=90)
        fig_left, fig_bottom, fig_width, fig_height = .15, .15, .75, .85
        axim = plt.axes([fig_left, fig_bottom, fig_width, fig_height])
        # self.implot = plt.imshow(self.image, cmap='gray', animated=True, interpolation='none')
        self.implot = plt.imshow(self.image, cmap='gray', animated=True,
                                 interpolation='none')
        axim.set_xticks([])
        axim.set_yticks([])
        self.xlim = self.figure.axes[0].get_xlim()
        self.ylim = self.figure.axes[0].get_ylim()
        self.axis = self.figure.get_axes()[0]
        # plot the circle center
        line = plt.plot([-1], [-1], '+', color=red)
        self.marks = [line[0]]
        # make a circle for each radius stored
        self.circles = []
        for radius, color in zip(self.data[['radius_small', 'radius_large']][0], [green, blue]):
            circle = plt.Circle(self.data[['x', 'y']][0], radius, color=color, fill=False)
            self.axis.add_patch(circle)
            self.circles += [circle]
        self.marker_lines_ax = self.figure.axes[0]
        self.marker_lines_ax.set_xlim(*self.xlim)
        self.marker_lines_ax.set_ylim(*self.ylim)
        self.image_data = self.axis.images[0]
        # title
        self.title = self.figure.suptitle(
            f"frame #{self.curr_frame_index + 1}")
        self.slider_ax = plt.axes([fig_left, 0.04, fig_width, 0.02])
        self.curr_frame = Slider(
            self.slider_ax, 'frame', 1, self.num_frames, valinit=1, valfmt='%d', color='k')
        self.curr_frame.on_changed(self.change_frame)
        # connect some keys
        self.cidk = self.figure.canvas.mpl_connect(
            'key_release_event', self.on_key_release)
        # change the toolbar functions
        NavigationToolbar2.home = self.show_image
        NavigationToolbar2.save = self.save_data
        # remove some default keyboard shortcuts
        try:
            plt.rcParams['keymap.pan'].remove('p')
            plt.rcParams['keymap.save'].remove('s')
            plt.rcParams['keymap.fullscreen'].remove('f')
        except:
            pass
        # make a list of objects and filenames to save
        self.objects_to_save = {}
        self.objects_to_save[self.tracking_fn] = self.data
        # plot markers
        self.playing = False
        self.show_image()
        # input box for setting the framerate
        self.framerate = 30
        # make the following buttons:
        # save
        self.save_button_ax = plt.axes([0.01, .30, .06, .05])
        self.save_button = Button(self.save_button_ax, 'Save')
        self.save_button.on_clicked(self.save_data)
        # play/pause
        self.play_pause_button_ax = plt.axes([0.07, .30, .06, .05])
        self.play_pause_button = Button(self.play_pause_button_ax, 'Play')
        self.play_pause_button.on_clicked(self.play)
        # the small radius slider
        small_radius_frame = plt.axes([fig_left + fig_width + .02, 0.1, .02, .05 + .7])
        self.small_radius_slider = Slider(
            small_radius_frame, 'small', 0, max(self.image.shape)/2,
            valinit=self.small_radius[0], valfmt='%d', color='k', orientation='vertical')
        self.small_radius_slider.on_changed(self.set_small_radius)
        # the large radius slider
        large_radius_frame = plt.axes([fig_left + fig_width + .07, 0.1, .02, .05 + .7])
        self.large_radius_slider = Slider(
            large_radius_frame, 'large', 0, max(self.image.shape)/2,
            valinit=self.large_radius[0], valfmt='%d', color='k', orientation='vertical')
        self.large_radius_slider.on_changed(self.set_large_radius)

    def set_small_radius(self, new_radius):
        self.small_radius[0] = float(new_radius)
        self.show_image()

    def set_large_radius(self, new_radius):
        self.large_radius[0] = float(new_radius)
        self.show_image()

    def load_file(self):
        if '.mat' in self.filename:
            matfile = scipy.io.loadmat(self.filename)
            self.video = matfile['vidData'].astype('int16')[:, :, 0]
            self.video = self.video.transpose(2, 0, 1)
            self.times = matfile['t_v'][:, 0]
        else:
            self.video = np.squeeze(io.vread(self.filename, as_grey=True)).astype('int16')
        # self.video = self.video.transpose((0, 2, 1))

    def toggle_hide(self, event=None):
        self.hide_others = not self.hide_others
        if self.hide_others:
            self.hide_button.label.set_text("Unhide")
        else:
            self.hide_button.label.set_text("Hide")
        self.hide_unhide_markers()

    def load_image(self):
        print(self.curr_frame_index)
        # print(len(self.filenames))
        self.image = self.video[self.curr_frame_index]
        # self.image = PIL.Image.open(self.filenames[self.curr_frame_index])
        # self.image = np.asarray(self.image)

    def show_image(self, *args):
        print('show_image')
        # first plot the image
        self.im = self.image
        # self.figure.axes[0].get_images()[0].set_data(self.im)
        self.implot.set_data(self.im)
        # then plot the markers
        for num, mark in enumerate(self.marks):
            mark.set_xdata(self.data['x'])
            mark.set_ydata(self.data['y'])
        # plot the two circles
        for num, (circle, radius) in enumerate(
                zip(self.circles, [self.small_radius, self.large_radius])):
            circle.center = self.center[0]
            circle.set_radius(radius[0])
        # and the title
        self.title.set_text(f"{self.filename} - frame #{self.curr_frame_index + 1}")
        self.figure.canvas.draw()
        plt.draw()

    def set_radius(self, text):
        radius = float(text)
        self.radii[self.curr_radius] = radius
        self.show_image()

    def set_framerate(self, text):
        num = float(text)
        self.framerate = num

    def playing_thread(self):
        # setup seems to have an inherent framerate of 10 fps
        # let's convert desired framerate to something near 10 fps but with desired jumps
        step = self.framerate / self.inherent_fps
        inherent_fps = self.framerate / step
        # self.animated_show()
        self.animation = FuncAnimation(self.figure, self.animated_show,
                                       interval=1000*self.framerate**-1,
                                       blit=False, repeat=True)
        self.animation.event_source.start()
        # breakpoint()
        # while self.playing:
        #     pass
        # self.animation.event_source.stop()
        # while self.playing:
        #     self.curr_frame.set_val(
        #         np.mod(self.curr_frame_index + 1 + step, self.num_frames))
        #     self.show_image()
        #     time.sleep(inherent_fps ** -1)

    def animated_show(self, event=None):
        self.curr_frame_index += 2
        self.curr_frame_index = np.mod(self.curr_frame_index, self.num_frames)
        self.curr_frame.set_val(self.curr_frame_index)
        # self.image = self.video[self.curr_frame_index - 1]
        # self.im = self.image
        self.implot.set_data(self.video[self.curr_frame_index - 1])
        # and the title
        self.title.set_text(f"{self.filename} - frame #{self.curr_frame_index + 1}")
        # plt.draw()
        # time.sleep(self.framerate**-1)
        
    # def play(self, event=None):
    #     if self.playing:
    #         self.animation.event_source.stop()
    #         del self.animation 
    #         self.playing = False
    #         self.load_image()
    #         self.play_pause_button.label.set_text("Play")
    #     else:
    #         self.animation = FuncAnimation(self.figure, self.animated_show,
    #                                        interval=1000*self.framerate**-1,
    #                                        blit=False, repeat=True)
    #         self.animation.event_source.start()
    #         # self.animated_show()
    #         # self.player = threading.Thread(target=self.playing_thread)
    #         # self.player.start()
    #         self.playing = True
    #         self.play_pause_button.label.set_text("Pause")

    def play(self, event=None):
        if self.playing:
            self.animation.event_source.stop()
            del self.animation 
            self.playing = False
            self.load_image()
            self.play_pause_button.label.set_text("Play")
        else:
            self.animation = FuncAnimation(self.figure, self.animated_show,
                                           # interval=1000*self.framerate**-1,
                                           interval=40,
                                           blit=False, repeat=True)
            self.animation.event_source.start()
            # self.animated_show()
            # self.player = threading.Thread(target=self.playing_thread)
            # self.player.start()
            self.playing = True
            self.play_pause_button.label.set_text("Pause")

    def change_frame(self, new_frame):
        if not self.playing:
            print('change_frame {} {}'.format(new_frame, int(new_frame)))
            self.curr_frame_index = int(new_frame)-1
            self.load_image()
            self.show_image()
            if self.data_changed:
                self.data_changed = False

    def nudge(self, direction):
        self.markers[self.curr_marker, 0, 0] += direction.real
        self.markers[self.curr_marker, 0, 1] += direction.imag
        self.show_image()
        # self.change_frame(mod(self.curr_frame, self.num_frames))
        self.data_changed = True

    def on_key_release(self, event):
        # frame change
        if event.key in ("pageup", "alt+v", "alt+tab", "p"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index, self.num_frames))
        elif event.key in ("pagedown", "alt+c", "tab", "n"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 2, self.num_frames))
            print(self.curr_frame_index)
        elif event.key == "alt+pageup":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index - 9, self.num_frames))
        elif event.key == "alt+pagedown":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 11, self.num_frames))
        elif event.key == "home":
            self.curr_frame.set_val(1)
        elif event.key == "end":
            self.curr_frame.set_val(self.num_frames)

        # if space pressed, set the marker there
        elif event.key == ' ':
            if event.inaxes == self.axis:
                self.center[0] = (event.xdata, event.ydata)
                self.data_changed = True
                self.show_image()
        # special keys
        elif event.key == "p":
            self.play()
        
        elif event.key == "backspace":
            self.data_changed = True
            self.data[:2] = -1
            self.show_image()

        # marker move
        elif event.key == "left":
            self.nudge(-1)
        elif event.key == "right":
            self.nudge(1)
        elif event.key == "up":
            self.nudge(-1j)
        elif event.key == "down":
            self.nudge(1j)
        elif event.key == "alt+left":
            self.nudge(-10)
        elif event.key == "alt+right":
            self.nudge(10)
        elif event.key == "alt+up":
            self.nudge(-10j)
        elif event.key == "alt+down":
            self.nudge(10j)
            
    def update_sliders(self, val):
        self.show_image()

    def on_mouse_release(self, event):
        self.center = [event.xdata, event.ydata]
        # self.change_frame(0)

    def save_data(self, event=None):
        print('save')
        for fn, val in zip(self.objects_to_save.keys(), self.objects_to_save.values()):
            np.save(fn, val)

if __name__ == "__main__":
    file_UI = FileSelector()
    file_UI.close()
    fn = file_UI.files[0]
    tracker = VideoTrackerWindow(fn)
    plt.show()

# # get the center and radii for all .mat files in the current folder
# fns = os.listdir("./")
# mat_fns = [fn for fn in fns if fn.endswith('.mat')]
# track_fns = os.listdir("tracking_data")
# for fn in mat_fns:
#     subject = ".".join(fn.split(".")[:-1])
#     if not any([subject in fn for fn in track_fns]):
#         tracker = VideoTrackerWindow(fn)
#         plt.show()

