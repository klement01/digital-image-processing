#!/usr/bin/env python3
"""Performs a forwards and reverse 2D DCT on an image."""
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
        help='reshapes the transform into a square')
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
    parser.add_argument(
        '-m',
        metavar='matrix_size',
        dest='matrix_size',
        type=int,
        default=0,
        help='the size of the transformation matrix')

    args = parser.parse_args()
    
    if args.grayscale:
        original = cv.imread(args.in_path, cv.IMREAD_GRAYSCALE)
    else:
        original = cv.imread(args.in_path)
    
    transf_org = fdct(original, args.matrix_size)
    
    transf = prettify(transf_org.copy())
    if args.square:
        transf = square(transf)
    cv.imwrite(args.out_path, transf)
    
    if args.recon_path is not None:
        reconstructed = idct(transf_org, args.matrix_size)
        cv.imwrite(args.recon_path, reconstructed)


def get_transform_matrix(M: int) -> (np.array, np.array):
    """Returns M x M forward DCT transform matrix."""
    t = np.full((M, M), 2, dtype=np.float64)
    t = (t * np.arange(M) + 1) * np.arange(M).reshape(M, 1)
    t = np.cos(t * np.pi / (2 * M))
    
    t *= np.sqrt(2 / M)
    
    c = np.ones((M, M), dtype=np.float64)
    c[0] /= np.sqrt(2)
    t *= c
    
    return t


def dct_1d(g: np.array) -> np.array:
    """Performs a 1D forward DTC."""
    if g.ndim != 1:
        raise ValueError("Array must be 1D for 1D DCT")
    
    M = len(g)
    s = np.sqrt(2 / M)
    G = np.empty_like(g, dtype=np.float64)
    
    for m in range(M):
        cm = 1
        if m == 0:
            cm = 1 / np.sqrt(2)
        
        acc = 0
        for u, gu in enumerate(g):
            phi = np.pi * m * (2 * u + 1) / (2 * M)
            acc += gu * cm * np.cos(phi)
            
        G[m] = s * acc
    
    return G


def idct_1d(G: np.array) -> np.array:
    """Performs a 1D reverse DTC."""
    if G.ndim != 1:
        raise ValueError("Array must be 1D for 1D inverse DCT")
    
    M = len(G)
    s = np.sqrt(2 / M)
    g = np.empty_like(G, dtype=np.float64)
 
    for u in range(M):
        acc = 0
        for m, Gm in enumerate(G):
            cm = 1
            if m == 0:
                cm = 1 / np.sqrt(2)
            
            phi = np.pi * m * (2 * u + 1) / (2 * M)
            acc += Gm * cm * np.cos(phi)
    
        g[u] = s * acc
    
    return g


def dct(g: np.array, forward: bool) -> np.array:
    """Performs a forward or inverse DTC on an image."""
    if g.ndim == 3:
        channels = g.transpose(2, 0, 1)
        channels_dct = np.array([dct(c, forward) for c in channels])
        return channels_dct.transpose(1, 2, 0)
    elif g.ndim != 2:
        raise ValueError("Invalid image")
        
    if forward:
        f = dct_1d
    else:
        f = idct_1d
        
    g = np.apply_along_axis(f, 0, g)
    g = np.apply_along_axis(f, 1, g)
    return g


def crop_matmul(A: np.array, B: np.array) -> np.array:
    """Performs ABAt', where B' is a chunk of B the same size as A,
    and At is the transpose of A."""
    if A.ndim != 2:
        raise ValueError("Invalid transform")
    
    if B.ndim == 3:
        channels = B.transpose(2, 0, 1)
        channels_dct = np.array([crop_matmul(A, c) for c in channels])
        return channels_dct.transpose(1, 2, 0)
    elif B.ndim != 2:
        raise ValueError("Invalid image")
    
    Bs, As = np.array([B.shape, A.shape])
    
    q = Bs // As
    
    vn, hn = q
    vs, hs = As * q
    B = B[:vs, :hs]
    
    split = np.split(B, vn) 
    split = [np.split(i, hn, 1) for i in split]
    
    split = [[np.matmul(A, i) for i in j] for j in split]
    split = [[np.matmul(i, A.transpose()) for i in j] for j in split]
    
    return np.block(split)


def fdct(g: np.array, matrix_size: int = 0) -> np.array:
    """Performs a forward DTC on an image."""
    if matrix_size > 0:
        A = get_transform_matrix(matrix_size)
        G = crop_matmul(A, g)
        
    else:
        G = dct(g, True)
    
    return G


def idct(G: np.array, matrix_size: int = 0) -> np.array:
    """Performs a reverse DTC on an image."""
    if matrix_size > 0:
        At = get_transform_matrix(matrix_size).transpose()
        g = crop_matmul(At, G)
        
    else:
        g = dct(G, True)
    
    return g


def prettify(g: np.array) -> np.array:
    """Filters the image for better visualization."""
    g = abs(g) + 1
    g = np.log(g)
    
    g -= g.min()
    g *= (2**16 - 1) / g.max()
    
    g = g.astype(np.uint16)
    
    if g.ndim == 3:
        g = cv.cvtColor(g, cv.COLOR_BGR2GRAY)

    return g


def square(g: np.array) -> np.array:
    """Turns the image into a square shape."""
    size = min(g.shape[:2])
    size = (size, size)
    return cv.resize(g, size)


if __name__ == '__main__':
    main()
