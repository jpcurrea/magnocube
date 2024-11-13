import cProfile
import pstats
import re
# 1) use cProfile to profile the run_video_mode_phone.py script and store the output to a file
# cProfile.run('re.compile("run_video_mode_phone.py")', 'phone_mode_profile')
# 2) use pstats to analyze the output
results = pstats.Stats('phone_profiler_stats.txt')
results.strip_dirs().sort_stats('cumulative').print_stats(100)