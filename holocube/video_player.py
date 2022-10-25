
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import sys
import subprocess



class VideoGUI(QtWidgets.QMainWindow):
    def __init__(self, display_rate=30):
        # make a QApplication
        # self.app = pg.mkQApp()
        self.display_interval = 1000. / display_rate
        super().__init__()
        # set the title
        self.setWindowTitle("Live Feed")
        # set the geometry
        self.setGeometry(100, 100, 600, 500)
        # setup the components
        self.setup()
        # show all the widgets
        self.show()

    def setup(self):
        # create an empty widget
        self.widget = QtWidgets.QWidget()
        # labels
        # configure
        pg.setConfigOptions(antialias=True)
        # create a graphics layout widget
        self.window = pg.GraphicsLayoutWidget()
        # add a view box
        self.view = self.window.addViewBox()
        # lock the aspec ratio
        self.view.setAspectLocked(True)
        # add the image item to the viewbox
        self.image = pg.ImageItem(border='k')
        self.view.addItem(self.image)
        # create a grid layout
        self.layout = QtWidgets.QGridLayout()
        # add layout to the main widget
        self.widget.setLayout(self.layout)
        # add the window and any other widgets to the layout
        self.layout.addWidget(self.window, 0, 1, 3, 1)
        # set widget as main widget
        self.setCentralWidget(self.widget)
        # run something
        self.first_frame = True

    def update_frame(self, frame):
        self.image.setImage(frame)
        if self.first_frame:
            # set the view bounds
            self.height, self.width = frame.shape[:2]
            self.view.setRange(QtCore.QRectF(0, 0, self.height, self.width))
            self.first_frame = False

class ValUpdater():
    def __init__(self, arr, display_rate=30.):
        self.interval = int(round(1000./display_rate))
        self.arr = arr
        self.ind = 0
        self.gui = VideoGUI()
        # setup the stdin reader
        # for frame in iter(sys.stdin.readline):
        # for frame in iter(sys.stdin):
        #         self.gui.update_frame(frame)
        # sys.stdin.flush()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.interval)
        self.timer.timeout.connect(self.update_val)
        self.timer.start()
        # start reading from the stdout

    def update_val(self):
        # todo: grab packets of data from the main loop using stdin
        val = self.arr[self.ind]
        # val = self.read_stdin()
        self.gui.update_frame(val)
        self.ind += 1
        self.ind %= len(self.arr)

    def read_stdin(self):
        """Listen for a packet of data from STDIN."""
        # todo: read from the stdout
        return None

if __name__ == "__main__":
    # get optional arguments
    # arg_list = np.array(sys.argv[1:])
    # # check for the height and width arguments
    # arg_dict = {}
    # for key, var in zip(['-h', '-w'], ['height', 'width']):
    #     if key in arg_list:
    #         ind = np.where(arg_list == key)[0]
    #         arg_dict[var] = int(arg_list[ind + 1])
    # get the
    app = pg.mkQApp()
    # vals = np.random.randint(
    #     0, 255, size=(1000, arg_dict['width'], arg_dict['height']))
    vals = np.random.randint(
        0, 255, size=(1000, 480, 640))
    animation = ValUpdater(vals)
    sys.exit(app.exec_())