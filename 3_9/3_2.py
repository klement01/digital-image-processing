#!/usr/bin/env python3
"""Generates the cumulative histogram of an image."""
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def main():
    parser = argparse.ArgumentParser(
        description='Generates the cumulative histogram of an image.')
    parser.add_argument(
        '--hist',
        dest='create_regular_histogram',
        action='store_true',
        help='if used, creates a regular histogram '
            'instead of a cumulative histogram')
    parser.add_argument(
        'in_image',
        metavar='in',
        help='the image to be processed')
    parser.add_argument(
        'out_image',
        metavar='out',
        nargs='?',
        default=None,
        help='where the histogram will be saved; '
            'if missing, simply displays the plot')
    args = parser.parse_args()
    
    image = cv.imread(args.in_image, cv.IMREAD_GRAYSCALE)
    if args.create_regular_histogram:
        c_hist, bin_edges = histogram_8bit_grayscale(image)
    else:
        c_hist, bin_edges = cumulative_histogram_8bit_grayscale(image)
    fig, _ = custom_histogram_plot(c_hist, bin_edges)
    if args.out_image is None:
        plt.show()
    else:
        fig.savefig(args.out_image)


def histogram_8bit_grayscale(image: np.array) -> (np.array, np.array):
    """Returns the histogram and list of bin edges for an array,
    optimized for 8-bit grayscale images."""
    return np.histogram(image, bins=256, range=(0,256))


def cumulative_histogram_8bit_grayscale(image: np.array) -> (np.array, np.array):
    """Returns the cumulative histogram and list of bin edges for an array,
    optimized for 8-bit grayscale images."""
    #Regular histogram.
    hist, bin_edges = histogram_8bit_grayscale(image)
    
    #Cumulative histogram based on regular histogram.
    c_hist = hist.cumsum()
    
    return c_hist, bin_edges


def custom_histogram_plot(data: np.array, bin_edges: np.array) -> (Figure, Axes):
    """Returns a custom bar plot optimized for histograms of cumulative
    histograms."""
    #Preprocesses the data
    data_x = bin_edges[:-1]
    data_y = data
    
    #Formatting constants.
    BLACK = '#000000'
    GRAY = '#3f3f3f'
    SCALE = 1.05
    LIM_X = data_x.max() * SCALE
    LIM_Y = data_y.max() * SCALE
    FONTFAMILY = 'serif'
    ARROWPROPS = dict(arrowstyle='->')
    ASPECT = 5/12
    
    #Creates the subplot for the plot and the plot itself.
    fig, ax = plt.subplots()
    ax.bar(
        data_x,
        data_y,
        width=1,
        align='edge',
        color=GRAY)
    
    #Labels and ticks.
    ax.set_xlabel('i', loc='right', fontfamily=FONTFAMILY)
    ax.set_ylabel('H(i)', loc='top', fontfamily=FONTFAMILY, rotation='horizontal')
    ax.set_xticks([bin_edges[0], bin_edges[-2]])
    ax.set_yticks([])
    ax.set_xlim(right=LIM_X)
    ax.set_ylim(top=LIM_Y)
    ax.set_box_aspect(ASPECT)
    
    #Frame and arrows.
    ax.set_frame_on(False)
    ax.annotate('', xy=(0, LIM_Y), xytext=(0, 0), arrowprops=ARROWPROPS)
    ax.annotate('', xy=(LIM_X, 0), xytext=(0, 0), arrowprops=ARROWPROPS)
    
    return fig, ax


if __name__ == '__main__':
    main()
