#!/usr/bin/env python3
"""Reconstructs an image from its first order integral."""
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
    parser.add_argument(
        'out_image',
        metavar='out',
        help='where the reconstructed image will be saved')
    args = parser.parse_args()
    
    integral = cv.imread(args.in_image, cv.IMREAD_GRAYSCALE)
    image = reconstruct_from_integral(integral)
    cv.imwrite(args.out_image, image)
    

def reconstruct_from_integral(integral: np.array) -> np.array:
    """Returns the image that generated the integral image."""
    integral = integral.astype(np.dtype('uint64'))
    
    flipped_integral = np.flipud(integral)
    flipped_integral = np.fliplr(flipped_integral)
    
    flipped_negative = np.diff(flipped_integral, axis=0, append=0)
    flipped_negative = np.diff(flipped_negative, axis=1, append=0)
    
    flipped_image = -flipped_negative
    
    image = np.flipud(flipped_image)
    image = np.fliplr(image)
    
    image = image.astype(np.dtype('uint8'))
    return image


if __name__ == '__main__':
    main()
