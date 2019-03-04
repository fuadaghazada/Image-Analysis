'''
    Implementation of spatial filters

    @author: Fuad Aghazada
    @date: 4/3/2019
    @version: 1.0
'''

import numpy as np

'''
    Convolve operation
'''
def convole(source_image, filter):

    h_s = filter.shape[0]    # Height of structuring element
    w_s = filter.shape[1]    # Width of structuring element

    # Output image
    out_img = np.zeros(source_image.shape).astype(np.uint8)


    # Vertical Indices
    i1 = int((h_s - 1) / 2)
    i2 = int(source_image.shape[0] - 1 - i1)

    # Horizontal Indices
    j1 = int((w_s - 1) / 2)
    j2 = int(source_image.shape[1] - 1 - j1)


    # Main loop for iterating source image
    for i in range(i1, i2 + 1):
        for j in range(j1, j2 + 1):
            value = average(i, j, source_image[i - i1: i + i1 + 1, j - j1: j + j1 + 1], filter)
            out_img[i, j] = value

    return out_img


'''
    Spatial linear averaging
'''
def average(i, j, source_image, filter):

    src_flat = source_image.flatten()
    fltr_flat = filter.flatten()

    sum = 0
    for i in range(0, len(fltr_flat)):
        sum += src_flat[i] * fltr_flat[i]

    return int(sum / len(fltr_flat))
