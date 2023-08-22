from tracking import *
from magno_tracker.tracking import *
import re

tracker = OfflineTracker('imported')
tracker.process_vids(gui=False, start_over=True)

