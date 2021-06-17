#!/usr/bin/env python3
"""Mirror an image horizontally, vertically, or both."""
import argparse
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Mirror an image.")
    parser.add_argument(
        'operation',
        metavar='operation',
        choices=['horizontal', 'vertical', 'both'],
        help='which mirroring operation will be performed')
    parser.add_argument(
        'in_image',
        metavar='in',
        help='the image to be processed')
    parser.add_argument(
        'out_image',
        metavar='out',
        help='where the processed image will be saved')
    args = parser.parse_args()
    
    image = Image.open(args.in_image)
    if args.operation in ('horizontal', 'both'):
        image = mirror_horizontal(image)
    if args.operation in ('vertical', 'both'):
        image = mirror_vertical(image)
    image.save(args.out_image)


def mirror_horizontal(image: Image) -> Image:
    """Returns a new image equal to the horizontal mirror of the input."""
    # Using Pillow methods:
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def mirror_vertical(image: Image) -> Image:
    """Returns a new image equal to the vertical mirror of the input."""
    # Using Pillow methods:
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


if __name__ == '__main__':
    main()
