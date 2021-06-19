#!/usr/bin/env python3
"""Sums and displays the pixel values of a grayscale image."""
import argparse
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'in_image',
        metavar='in',
        help='the image to be processed')
    args = parser.parse_args()
    
    image = Image.open(args.in_image)
    result = count_values(image)
    print(f"Result: {result}")


def count_values(image: Image) -> int:
    """Returns the sum of all pixel values in a grayscale image."""
    #Converts the image to 8-bit grayscale for processing.
    image = image.convert("L")
    
    #Sums the values.
    result = sum(list(image.getdata()))
    
    return result


if __name__ == '__main__':
    main()
