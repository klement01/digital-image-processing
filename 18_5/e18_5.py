#!/usr/bin/env python3
"""Performs a forwards and reverse 1D Fourier transform on an image."""
import argparse
import cv2 as cv
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
    
    original = cv.imread(args.in_image, cv.IMREAD_GRAYSCALE)
    fourier = DFT(original, True)
    reconstructed = DFT(fourier, False).reshape(original.shape)
    reconstructed = complex_to_grayscale(reconstructed)
    cv.imwrite(args.out_image, reconstructed)


def DFT(g: np.array, forward: bool) -> np.array:
    """Performs a 1D Discrete Fourier Transform on a series of complex values.
    
    Based on program 18.1 from Digital Image Processing by Wilhelm Burger and
    Mark J. Burge."""
    g = g.ravel()
    M = len(g)
    G = np.empty_like(g, dtype=complex)
    s = 1 / np.sqrt(M)
    
    def cos_sin_cache():
        """Returns the appropriate cosine and sine for a value of m using
        a cache."""
        cache = {}
        mu = yield
        while True:
            k = mu % M
            if k in cache:
                mu = yield cache[k]
            else:
                a = 2 * np.pi * k / M
                c = np.cos(a)
                s = np.sin(a)
                cache[k] = (c, s)
                mu = yield c, s
    cs = cos_sin_cache()
    next(cs)
    
    for m in range(M):
        total = complex(0, 0)
        for u, item in enumerate(g):
            cosw, sinw = cs.send(m * u)
            if not forward:
                sinw = -sinw
            transform = complex(cosw, sinw)
            total += item * transform
        G[m] = total * s
    
    return G


def complex_to_grayscale(g: np.array) -> np.array:
    """Turns an array of complex numbers into a grayscale image."""
    def ctgs(p):
        v = round(abs(p))
        v = max(v, 0)
        v = min(v, 255)
        return v
    return np.vectorize(ctgs, otypes=['uint8'])(g)


if __name__ == '__main__':
    main()
