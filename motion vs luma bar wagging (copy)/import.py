import os
from tracking import *

if not os.path.isdir("imported"):
    os.mkdir("imported")

# combine inputs by adding a genotype attribute
control_exp = TrackingExperiment('Empty Sp Gal4', remove_incompletes=False)
kir_exp = TrackingExperiment('SS00096 UAS Kir2.1', remove_incompletes=False)
for exp in [control_exp, kir_exp]:
    for trial in exp.trials:
        trial.add_attr('dirname', exp.dirname)
        # now move the file to the imported folder
        for fn in [trial.filename, trial.video_file]:
            new_fn = os.path.join("imported", os.path.basename(fn))
            os.rename(fn, new_fn)
            # print progress
            print(new_fn)