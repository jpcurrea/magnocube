# %% [markdown]
# # The Effect of the Heading Compass in Optomotor and Saccadic Motion Tracking
# 
# We are trying to understand why flies are more likely to saccade when there is a moving motion-defined bar rather than a contrast defined bar. To do this, we are going to wiggle one of either case (motion vs dark or light bar on a gray background). We know from previous pilotperiments that flies will orient towards a luminance defined bar when there is no background, but won't if there is a textured background. A wiggling bar in the absence of a background seems to engage both the object and optomotor response systems at the same time, resulting in this orientation behavior. However, we should first demonstrate the fundamental observation made through discussions with the Damon Clark lab.
# 
# Moreover, this specific analysis will look at the effect of EPG neurons in the central complex. These are supposedly required for the bump to propogate, so we are effectively studying the role of updating the heading compass on optomotor and saccadic responses to high spatial frequency information.
# 
# So here are my parameters for this experiment:
# 
# 0) genotype: Empty Split Gal4 vs. EPG-Kir
# 1) background: closed-loop present vs. absent (x2)
# 2) bar type: dark, light, or motion-defined (x3)
# 3) hemisphere: left vs right (x2)
# total= 2 * 3 * 2 tests = 12 tests

# %%
"""Import the dataset and do some pre-processing."""
from tracking import *

# import the experiment data
exp = TrackingExperiment('imported')
# offline tracking has already been done, so we can just use the offline data
# but the fidelity of online tracking is needed to know where the bar was
# so let's compare the online to offline tracking and just keep those with small differences
heading_online = exp.query(output='camera_heading', sort_by='test_ind')
heading_offline = exp.query(output='camera_heading_offline', sort_by='test_ind')
# 
print(heading_online.shape, heading_offline.shape)

# %%
res = dir(exp.trials[10])
print(res)


