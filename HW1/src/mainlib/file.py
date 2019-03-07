'''
    File image operations: loading and saving the images

    @author: Fuad Aghazada
    @date: 4/3/2019
    @version: 1.0
'''

import sys
import numpy as np
from PIL import Image


'''
    Loading Image as 2D array
'''
def load_img_2D(file_name):

    arr = None

    try:
        img = Image.open('../in/' + str(file_name)).convert('L')
        arr = np.asarray(img)

    except Exception as e:
        print(e)

    return arr

'''
    Saving Image from 2D array
'''
def save_img_from_array(arr, out_name = 'out_img.png'):

    out = None

    try:
        out = Image.fromarray(arr)
        out.save('../out/' + str(out_name))
    except Exception as e:
        raise

    return out
