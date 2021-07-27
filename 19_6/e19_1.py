#!/usr/bin/env python3
"""Performs a forwards and reverse 2D Fourier transform on an image."""
import argparse
import cv2 as cv
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
        'in_path',
        metavar='in',
        help='the image to be processed')
    parser.add_argument(
        'out_path',
        metavar='out',
        help='where the transform will be saved')
    parser.add_argument(
        '-s', '--square',
        action='store_true',
        dest='square',
        help='reshapes the Fourier transform into a square')
    parser.add_argument(
        '-g', '--grayscale',
        action='store_true',
        dest='grayscale',
        help='read the image in grayscale mode')
    parser.add_argument(
        '-r',
        metavar='recon',
        dest='recon_path',
        help='where the reconstructed image will be saved')

    args = parser.parse_args()
    
    if args.grayscale:
        original = cv.imread(args.in_path, cv.IMREAD_GRAYSCALE)
    else:
        original = cv.imread(args.in_path)
    
    fourier = fft(original)
    fourier_pt = fourier_prettify(fourier)
    if args.square:
        fourier_pt = square(fourier_pt)
    cv.imwrite(args.out_path, fourier_pt)
    
    if args.recon_path is not None:
        reconstructed = ifft(fourier)
        reconstructed_gs = complex_to_grayscale(reconstructed)
        cv.imwrite(args.recon_path, reconstructed_gs)


def nd_dft(g: np.array, forward: bool) -> np.array:
    """Computes the n-dimensional Discrete Fourier Transform of g."""
    f = np.fft.fft if forward else np.fft.ifft
    for i in range(g.ndim):
       g = f(g, axis=i)
    return g


def fft(g: np.array) -> np.array:
    """Computes the forward n-dimensional Discrete Fourier Transform of g."""
    return nd_dft(g, True)


def ifft(g: np.array) -> np.array:
    """Computes the inverse n-dimensional Discrete Fourier Transform of g."""
    return nd_dft(g, False)


def complex_to_grayscale(g: np.array) -> np.array:
    """Turns an array of complex numbers into a grayscale image."""
    g_abs = abs(g)
    g_normal = 255 * g_abs / g_abs.max()
    return np.array(g_normal, dtype='uint8')


def half(ary, axis=0):
    """Splits an array in half in an axis."""
    return np.array_split(ary, 2, axis=axis)


def vhalf(ary):
    """Splits an array in half vertically (axis 0.)"""
    return half(ary, axis=0)


def hhalf(ary):
    """Splits an array in half horizontally (axis 1.)"""
    return half(ary, axis=1)


def fourier_prettify(g: np.array) -> np.array:
    """Modifies a 2D DCT so it can be better visualized as an image."""
    #Turns the values from complex to reals
    g = np.log(abs(g))
    
    #Reorganizes the quadrants of the image
    w, e = hhalf(g)
    quads = vhalf(w) + vhalf(e)
    w = np.vstack((quads[3], quads[2]))
    e = np.vstack((quads[1], quads[0]))
    g = np.hstack((w, e))
    
    #Normalizes the image  and turns it to grayscale
    g = complex_to_grayscale(g)
    g = cv.cvtColor(g, cv.COLOR_BGR2GRAY)

    return g


def square(g: np.array) -> np.array:
    """Turns the image into a square shape."""
    size = min(g.shape[:2])
    size = (size, size)
    return cv.resize(g, size)


if __name__ == '__main__':
    main()
