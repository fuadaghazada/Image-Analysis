'''
    Implementation of 2D discrete Fourier Transform

    @author: Fuad Aghazada
    @date: 5/3/2019
    @version: 1.0
'''

import cmath
import numpy as np

'''
    Forward 2D Discrete Fourier Transform
'''
def dft2D(source_image):

    # Image dimensions
    M = source_image.shape[0]
    N = source_image.shape[1]

    dft = np.zeros((M, N))

    for x in range(0, M):
        for y in range(0, N):
            value = 0
            for u in range(0, M):
                for v in range(0, N):
                    pixel = source_image[u, v]
                    value += pixel * cmath.exp(- 1j * cmath.pi * 2.0 * (float(x * u) / M + float(y * v) / N))
            dft[x, y] = value / (M * N)

    return dft


'''
    Backward (Inverse) 2D Discrete Fourier Transform
'''
def idft2D(dft):

    # Image dimensions
    M = dft.shape[0]
    N = dft.shape[1]

    source_image = np.zeros((M, N))

    for u in range(0, M):
        for v in range(0, N):
            value = 0
            for x in range(0, M):
                for y in range(0, N):
                    pixel = dft[x, y]
                    value += pixel * cmath.exp(1j * cmath.pi * 2.0 * (float(x * u) / M + float(y * v) / N))
            source_image[u, v] = int(value)

    return source_image
