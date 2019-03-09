'''
    Implementation of filtering in frequency domain

    @author: Fuad Aghazada
    @date: 5/3/2019
    @version: 1.0
'''

import numpy as np
import math

'''
    Filtering in frequency domain: Gaussian filter
'''
def gaussian_filter(source_image, sigma = 0.2):

    (M, N) = source_image.shape

    # Calculating gaussian kernel
    g_func = np.ones(source_image.shape)
    for i in range(0, M):
        for j in range(0, N):
            g_func[i, j] = math.exp(-(i * i + j * j) / (2 * sigma * sigma)) / (sigma * sigma * math.pi * 2.0)

    # Normalizing
    g_func /= np.sum(g_func)

    # Fourier transform of source_image & filter
    fft_src_img = np.fft.fft2(source_image)
    fft_filter_func = np.fft.fft2(g_func)

    fft_out = np.multiply(fft_src_img, fft_filter_func)
    out = np.fft.ifft2(fft_out)
    out = out.real

    return np.log(fft_src_img).real, g_func, np.log(fft_filter_func).real, out, np.log(fft_out).real
