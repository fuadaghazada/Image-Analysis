'''
    Implementation of 2 morphological operations: dilation and erosion +
    Implementation of thresholding
    First Notation has been used for implementing these morphological operations.

    @author: Fuad Aghazada
    @date: 4/3/2019
    @version: 1.0
'''


import numpy as np

'''
    Dilation operation
'''
def dilation(source_image, struct_el):

    out_img = np.zeros(source_image.shape).astype(np.uint8)

    # Main loop for iterating source image
    for i in range(0, len(source_image)):
        for j in range(0, len(source_image[i])):
            if check_notation(i, j, source_image, struct_el, 1) is True:
                out_img[i, j] = 255

    return out_img


'''
    Erosion operation
'''
def erosion(source_image, struct_el):

    out_img = np.zeros(source_image.shape).astype(np.uint8)

    # Main loop for iterating source image
    for i in range(0, len(source_image)):
        for j in range(0, len(source_image[i])):
            if check_notation(i, j, source_image, struct_el, 0) is True:
                out_img[i, j] = 255

    return out_img


'''
    Thresholding with the given value
'''
def threshold(img, value):

    img.setflags(write = 1)     # Setting write to 1

    img[img > value] = 255
    img[img <= value] = 0

    img.setflags(write = 0)     # Setting write back to 0


'''
    Checking the notation 1 for both operations - dilation and erosion:

    type:
        0: for dilation
        1: for erosion
'''
def check_notation(i, j, source_image, struct_el, type = 0):

    h_s = struct_el.shape[0]    # Height of structuring element
    w_s = struct_el.shape[1]    # Width of structuring element

    # Horizontal starting index
    a1 = int(j - (w_s - 1) / 2)
    if a1 < 0:
        a1 = 0

    # Horizontal ending index
    a2 = int(j + (w_s - 1) / 2)
    if a2 >= len(source_image[i]):
        a2 = len(source_image[i]) - 1

    # Vertical starting index
    b1 = int(i - (h_s - 1) / 2)
    if b1 < 0:
        b1 = 0

    # Vertical ending index
    b2 = int(i + (h_s - 1) / 2)
    if b2 >= len(source_image):
        b2 = len(source_image) - 1

    # Sliced part
    sliced_part = source_image[b1 : b2 + 1, a1 : a2 + 1]

    # Checking type
    result = False

    if type == 0:
        # Check commonality by set intersection
        a = set(sliced_part.flatten())
        b = set(struct_el.flatten())

        result = len(b.intersection(a)) >= 1
    else:
        # Check commonality by list intersection
        a = list(sliced_part.flatten())
        b = list(struct_el.flatten())

        result = (a == b)

    return result
