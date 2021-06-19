#!/usr/bin/env python3
"""Generates a random grayscale image and its histogram."""
import argparse
import cv2 as cv
import e3_2 as histogram
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
        '-r',
        dest='resolution',
        metavar=('X','Y'),
        nargs=2,
        type=int,
        default=[500, 500],
        help='the resolution of the generated image (defaut: %(default)s)')
    parser.add_argument(
        'out_hist',
        metavar='histogram',
        nargs='?',
        default=None,
        help='where the histogram will be saved; if missing, histogram will be shown')
    parser.add_argument(
        'out_image',
        metavar='image',
        nargs='?',
        default=None,
        help='where the image will be saved')
    args = parser.parse_args()
    
    image = random_grayscale_image(args.resolution)
    if args.out_image is not None:
        cv.imwrite(args.out_image, image)
    hist, bin_edges = histogram.histogram_8bit_grayscale(image)
    fig, _ = histogram.custom_histogram_plot(hist, bin_edges)
    if args.out_hist is not None:
        fig.savefig(args.out_hist)
    else:
        plt.show()


def random_grayscale_image(resolution: [int, int]) -> np.array:
    """Returns an 8-bit grayscale image with random colors and the
    specified resolution."""
    random = np.random.random_sample(resolution[::-1])
    random *= 256
    random = random.astype(np.dtype('uint8'))
    return random


if __name__ == '__main__':
    main()
