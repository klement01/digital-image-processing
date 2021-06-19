#!/usr/bin/env python3
"""Mirror an image horizontally, vertically, or both."""
import argparse
from enum import Enum, auto
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'operation',
        metavar='operation',
        choices=['horizontal', 'vertical', 'both'],
        help='which mirroring operation will be performed (%(choices)s)')
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


class MirrorMode(Enum):
    """Possible modes of operation for mirror_generic."""
    HORIZONTAL = auto()
    VERTICAL = auto()

def mirror_generic(image: Image, mirror_type: MirrorMode) -> Image:
    """Returns a new image that's mirrored vertically or horizontally"""
    #Generates the new data.
    w, h = image.size
    data = list(image.getdata())
    new_data = []
    for y in range(h):
        for x in range(w):
            if mirror_type is MirrorMode.HORIZONTAL:
                x2 = w - x - 1
                i = w * y + x2
            else:
                y2 = h - y - 1
                i = w * y2 + x
            new_data.append(data[i])

    #Flattens the new data if necessary and turn it into bytes.
    if type(new_data[0]) is tuple:
        new_data = bytes(c for p in new_data for c in p)
    else:
        new_data = bytes(new_data)

    new_image = Image.frombytes(image.mode, image.size, new_data)
    
    return new_image


def mirror_horizontal(image: Image) -> Image:
    """Returns a new image equal to the horizontal mirror of the input."""
    #Using Pillow methods (fast):
    #new_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    #Doing it manually (slow):
    new_image = mirror_generic(image, MirrorMode.HORIZONTAL)

    return new_image


def mirror_vertical(image: Image) -> Image:
    """Returns a new image equal to the vertical mirror of the input."""
    #Using Pillow methods (fast):
    #new_image = image.transpose(Image.FLIP_TOP_BOTTOM)

    #Doing it manually (slow):
    new_image = mirror_generic(image, MirrorMode.VERTICAL)
    
    return new_image


if __name__ == '__main__':
    main()
