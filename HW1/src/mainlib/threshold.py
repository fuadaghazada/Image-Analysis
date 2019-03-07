'''
    Implementation of global and adaptive thresholding

    @author: Fuad Aghazada
    @date: 4/3/2019
    @version: 1.0
'''

import numpy as np
from pprint import pprint

'''
    Thresholding with the given value
'''
def threshold(img, value = None):

    img.setflags(write = 1)     # Setting write to 1

    # Calculating threshold by sum / (n * m)
    if value is None:
        dimension = img.shape
        value = np.sum(img) / (dimension[0] * dimension[1])

    img[img >= value] = 255
    img[img < value] = 0

    img.setflags(write = 0)     # Setting write back to 0


'''
    Adaptive thresholding

    source_image:
        input image
    neighbours:
        local area of pixel for taking mean or median
    C:
        constant to be substracted from threshold
    mode:
        0 - for mean
        1 - for median
'''
def adaptive_threshold(source_image, neighbours = (3, 3), C = 3, mode = 0):

    (h_s, w_s) = neighbours

    # Vertical Indices
    i1 = int((h_s) / 2)
    i2 = int(source_image.shape[0] - 1 - i1)

    # Horizontal Indices
    j1 = int((w_s) / 2)
    j2 = int(source_image.shape[1] - 1 - j1)

    # Output image
    out_img = np.ones(source_image.shape).astype(np.uint8) * 255    # White img

    # Main loop for iterating source image
    for i in range(i1, i2 + 1):
        for j in range(j1, j2 + 1):
            if mode == 0:
                value = np.mean(source_image[i - i1: i + i1 + 1, j - j1: j + j1 + 1].flatten())
            else:
                value = np.median(source_image[i - i1: i + i1 + 1, j - j1: j + j1 + 1].flatten())

            value -= C  # Constant

            if source_image[i, j] >= value:
                out_img[i, j] = 255
            else:
                out_img[i, j] = 0

    return out_img
