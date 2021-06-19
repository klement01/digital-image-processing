#!/usr/bin/env python3
"""Creates a video of the image being shifted horizontally and circularly
until the original state is reached again."""
import argparse
import cv2 as cv
from numpy import concatenate, ndarray


def main():
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
        '--duration',
        dest='duration',
        type=float,
        default=10,
        help='how long the video should be, in seconds (default: %(default).1f)')
    parser.add_argument(
        'in_image',
        metavar='in',
        help='the image to be processed')
    parser.add_argument(
        'out_video',
        metavar='out',
        help='where the video will be saved')
    args = parser.parse_args()

    image = cv.imread(args.in_image)
    create_video(args.out_video, image, args.duration)


def shift_right(array: ndarray) -> ndarray:
    """Shifts the array one column to the right, wrapping around the column
    that gets pushed out."""
    new_array = array.swapaxes(0, 1)
    #Head and tail of the new array == tail and head of the original array.
    head = new_array[-1:]
    tail = new_array[:-1]
    new_array = concatenate([head, tail])
    new_array = new_array.swapaxes(0, 1)
    return new_array


def create_video(out_file: str, image: ndarray, duration: float, fps: float = 60.0) -> None:
    """Generates all video frames and writes them to the destination file."""
    #Creates a writer for the video.
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    writer = cv.VideoWriter(out_file, fourcc, fps, image.shape[1::-1])

    #Calculates the duration, in frames, of each image.
    #Repeats the initial state at the beginning and the end
    total_shifts = image.shape[1]
    total_frames = round(fps * duration)

    #Writes the video frame by frame, shifting the image along
    #the way based on the fraction of the video written.
    shift_count = 0
    for frame in range(total_frames):
        while (frame + 1) / total_frames > shift_count / total_shifts:
            shift_count += 1
            image = shift_right(image)
        writer.write(image)

    writer.release()


if __name__ == '__main__':
    main()
