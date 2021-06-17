#!/usr/bin/env python3
"""Finds and displays the minimum and maximum pixel values of a grayscale image."""
import argparse
from PIL import Image, ImageDraw
from collections import namedtuple


def main():
    parser = argparse.ArgumentParser(description='Finds and displays the minimum and maximum pixel values of a grayscale image.')
    parser.add_argument(
        'in_image',
        metavar='in',
        help='the image to be processed')
    args = parser.parse_args()
    
    image = Image.open(args.in_image)
    pair = find_min_max(image)
    print(f"Min: {pair.min}")
    print(f"Max: {pair.max}")


MinMaxPair = namedtuple("MinMaxPair", "min max")

def find_min_max(image: Image) -> MinMaxPair:
    """Returns the minimum and maximum pixel values of a grayscale image."""
    #Converts the image to 8-bit grayscale for processing.
    image = image.convert("L")
    
    #Finds the values.
    data = list(image.getdata())
    pixel_min = min(data)
    pixel_max = max(data)
    pair = MinMaxPair(pixel_min, pixel_max)
    
    return pair


if __name__ == '__main__':
    main()
