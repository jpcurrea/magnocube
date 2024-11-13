import os
from magno_tracker.tracking import *

exp = TrackingExperiment('mseq_test')
# grab the first trial
trial = exp.trials[0]
# get the light sensor measurements and the original m-sequence data
breakpoint()
daq_data = trial.analog
light_vals, mseq = daq_data[:]
# 