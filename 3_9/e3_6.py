#!/usr/bin/env python3
"""Calculates the mean and variance of a grayscale image."""
import argparse
import cv2 as cv
import e3_2 as histogram
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
        'in_image',
        metavar='in',
        help='the image to be processed')
    args = parser.parse_args()
    
    image = cv.imread(args.in_image, cv.IMREAD_GRAYSCALE)
    mean, var = grayscale_mean_var(image)
    print(f"Mean: {mean}")
    print(f"Variance: {var}")


def grayscale_mean_var(image: np.array) -> (np.float64, np.float64):
    """Returns the mean and variance of a grayscale image."""
    hist, bin_edges = histogram.histogram_8bit_grayscale(image)
    
    A = (hist * bin_edges[:-1]).sum()
    B = (hist * bin_edges[:-1] ** 2).sum()
    MN = hist.sum()
    
    mean = A / MN
    var = (B - A ** 2 / MN) / MN

    return mean, var


if __name__ == '__main__':
    main()
