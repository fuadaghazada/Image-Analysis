'''
    Implementation of filtering in spatial domain

    @author: Fuad Aghazada
    @date: 4/3/2019
    @version: 1.0
'''

import numpy as np
import statistics

from PIL import Image

'''
    Convolve operation

    mode:
        0 for sum
        1 for median
'''
def convole(source_image, filter, mode = 0):

    h_s = filter.shape[0]    # Height of filter element
    w_s = filter.shape[1]    # Width of filter element

    # Vertical Indices
    i1 = int((h_s - 1) / 2)
    i2 = int(source_image.shape[0] - 1 - i1)

    # Horizontal Indices
    j1 = int((w_s - 1) / 2)
    j2 = int(source_image.shape[1] - 1 - j1)

    # Output image
    out_img = Image.new('L', (source_image.shape[1], source_image.shape[0]))
    pixels = out_img.load()

    # Main loop for iterating source image
    for i in range(i1, i2 + 1):
        for j in range(j1, j2 + 1):
            if mode == 0:
                value = sum(i, j, source_image[i - i1: i + i1 + 1, j - j1: j + j1 + 1], filter)
            else:
                value = median(i, j, source_image[i - i1: i + i1 + 1, j - j1: j + j1 + 1], filter)

            pixels[j, i] = int(value)

    return out_img


'''
    Spatial linear averaging
'''
def sum(i, j, source_image, filter):

    src_flat = source_image.flatten()
    fltr_flat = filter.flatten()

    sum = 0
    for i in range(0, len(fltr_flat)):
        sum += int(src_flat[i]) * int(fltr_flat[i])

    return sum


'''
    Spatial median ordering
'''
def median(i, j, source_image, filter):

    src_flat = source_image.flatten()
    fltr_flat = filter.flatten()

    temp = []
    for i in range(0, len(fltr_flat)):
        temp.append(int(src_flat[i]) * int(fltr_flat[i]))

    return statistics.median(temp)
