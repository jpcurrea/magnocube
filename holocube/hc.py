# instantiate the classes needed for 5 sided holocube operation and
# set up the namespace for import by run and experiments

import holocube.windows as windows
import holocube.schedulers as schedulers
import holocube.stimuli as stim
import holocube.arduino as ard
import holocube.tools as tools
import holocube.camera as cameras
import holocube.multiplexer as multiplexer

# objects we need in run and exps
window = windows.Holocube_window()
scheduler = schedulers.Scheduler()
arduino = ard.Arduino()
try:
    camera = cameras.Camera(window=window, com_correction=False)
except:
    camera = None
try:
    multiplexer = multiplexer.Multiplexer()
except:
    multiplexer = None
