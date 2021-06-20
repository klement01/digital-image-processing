#!/usr/bin/env python3
"""Calculates the first order integral image of a grayscale image."""
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
        '-t', '--top',
        dest='top_left',
        metavar=('X','Y'),
        nargs=2,
        type=int,
        default=[0, 0],
        help='coordinates of top left vertex (default: %(default)s)')
    parser.add_argument(
        '-b', '--bot',
        dest='bottom_right',
        metavar=('X','Y'),
        nargs=2,
        type=int,
        default=None,
        help='coordinates of bottom right vertex (default: bottom right of image)')
    parser.add_argument(
        '-o', '--output',
        metavar='out',
        dest='out_image',
        default=None,
        help='where to save a visual representation of the result')
    args = parser.parse_args()
    
    image = cv.imread(args.in_image, cv.IMREAD_GRAYSCALE)
    integral = integral_image_grayscale(image)
    result = first_order_block_sum(integral, args.top_left, args.bottom_right)
    print(f"First order block sum: {result}")
    if args.out_image is not None:
        representation = integral_image_representation(
            integral, args.top_left, args.bottom_right)
        cv.imwrite(args.out_image, representation)


def integral_image_grayscale(image: np.array) -> np.array:
    """Returns the first integral image of a grayscale image."""
    integral = image.cumsum(axis=0)
    integral.cumsum(axis=1, out=integral)
    return integral


def first_order_block_sum(integral: np.array, top_left: [int, int] = [0, 0],
        bottom_right: [int, int] = None) -> np.array:
    """Returns the first order block sum of the region with the
    given coordinates."""
    def get_value(x, y):
        if x < 0 or y < 0:
            return 0
        return integral[y, x]
    
    if bottom_right is None:
        bottom_right = tuple(i - 1 for i in integral.shape[1::-1])
    
    top_x, top_y = top_left
    bot_x, bot_y = bottom_right
    
    A = get_value(top_x - 1, top_y - 1)
    B = get_value(bot_x,     top_y - 1)
    C = get_value(top_x - 1, bot_y)
    D = get_value(bot_x,     bot_y)

    R = D - C - B + A
    return R


def integral_image_representation(integral: np.array, top_left: [int, int] = [0, 0],
        bottom_right: [int, int] = None) -> np.array:
    """Returns a grayscale image representation of a region of an
    integral image."""
    if bottom_right is None:
        bottom_right = tuple(i - 1 for i in integral.shape[1::-1])
    
    top_x, top_y = top_left
    bot_x, bot_y = bottom_right
    
    image = integral[top_y : bot_y + 1, top_x : bot_x + 1]
    image = image.astype(np.dtype('float64'))
    image -= image.min()
    image *= 255 / image.max()
    image = image.astype(np.dtype('uint8'))
    return image


if __name__ == '__main__':
    main()
