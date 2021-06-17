#!/usr/bin/env python3
"""Paints a 10 x 10 pixels white frame in the top left corner of the image."""
import argparse
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description='Paints a 10 x 10 pixels white frame in the top left corner of the image.')
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
    image = paint_frame(image)
    image.save(args.out_image)


def paint_frame(image: Image) -> Image:
    """Returns a new image with a 10 x 10 pixels white frame painted on its top left corner."""
    #Converts the image to RGB for processing.
    original_mode = image.mode
    new_image = image.convert("RGB")
    
    #Paints the frame.
    draw = ImageDraw.Draw(new_image)
    box = (0, 0, 9, 9)
    color = (255, 255, 255)
    draw.rectangle(box, fill=color)
    
    #Converts the image back to the original mode and returns it.
    new_image = new_image.convert(original_mode)
    return new_image


if __name__ == '__main__':
    main()
