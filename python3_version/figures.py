#!/usr/bin/env python
# -- coding: utf-8 --

"""
Hannah Weiser
h.weiser@stud.uni-heidelberg.de
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def find_bins(min, max, width):
    bound_min = -1.0 * (min % width - min)
    bound_max = max - max % width + width
    n = int((bound_max - bound_min) / width) + 1
    bins = np.linspace(bound_min, bound_max, n)
    return bins


out_dir = r"I:\UAV-photo\FlowVeloTool_Befliegung2020_output7"
# out_dir = r"I:\UAV-photo\FlowVeloTool_Befliegung2020_output_test"

os.chdir(out_dir)
file_final_tracks = "TracksFiltered_PTV_VeloThresh.txt"
final_tracks = pd.read_csv(file_final_tracks, sep="\t")

# stats
velo_mean = np.mean(final_tracks["velo"])
velo_median = np.median(final_tracks["velo"])
velo_min = np.min(final_tracks["velo"])
velo_max = np.max(final_tracks["velo"])
velo_std = np.std(final_tracks["velo"])
n = final_tracks.shape[0]

print("n: %i\n"
      "mean: %.3f\n"
      "median: %.3f\n"
      "min: %.3f\n"
      "max: %.3f\n"
      "std: %.3f" % (n, velo_mean, velo_median, velo_min, velo_max, velo_std))

bin_size = 0.05
bins = find_bins(velo_min, velo_max, bin_size)

plt.hist(final_tracks["velo"], bins=bins)
plt.xlabel("Velocity [m/s]")
plt.ylabel("Track frequency")
plt.show()

