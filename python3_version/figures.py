#!/usr/bin/env python
# -- coding: utf-8 --

"""
Hannah Weiser
h.weiser@stud.uni-heidelberg.de

File to compute basic statistics from final filtered FlowVelo tracks and write a histogram with the statistics to
an output PNG-file (histo.png).

Execution (example)

figures.py --dir I:\UAV-photo\FlowVeloTool_output

To view the help, run:
figures.py --help
or
figures.py -h
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt


def find_bins(min_val, max_val, width):
    bound_min = -1.0 * (min_val % width - min_val)
    bound_max = max_val - max_val % width + width
    n = int((bound_max - bound_min) / width) + 1
    bins = np.linspace(bound_min, bound_max, n)
    return bins


def readUserArguments():
    # Generate help and user argument parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="(c) Hannah Weiser (2021) - Heidelberg University")
    parser.add_argument("--dir", dest='directory', type=str,
                        help="Output directory of the FlowVelo tool, containing the results of feature tracking, filtering and velocity computation",
                        required=True)
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':

    opts = readUserArguments()
    out_dir = opts.directory

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

    font = {'family': 'DejaVu Sans',
            'color':  'black',
            'weight': 'normal',
            'size': 12.5,
            }

    stats = """
    n             %i
    min         %.3f
    max        %.3f
    mean      %.3f
    median   %.3f
    std          %.3f
            """ % (n, velo_min, velo_max, velo_mean, velo_median, velo_std)

    bin_size = 0.025
    bins = find_bins(velo_min, velo_max, bin_size)

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.hist(final_tracks["velo"], bins=bins)
    plt.xlabel("Velocity [m/s]", fontdict=font)
    plt.ylabel("Track frequency", fontdict=font)
    bottom, top = ax.get_ylim()
    yloc = bottom + top/5 * 3
    left, right = ax.get_xlim()
    plt.text(left, yloc, stats, fontdict=font)
    plt.savefig("histo.png", dpi=100)
