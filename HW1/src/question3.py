from pprint import pprint
import numpy as np

from mainlib.file import load_img_2D, save_img_from_array
from mainlib.threshold import threshold
from mainlib.sfiltering import convole

from other.loading import execute

######## QUESTION 3 ########

def question3_test():

    src_img = load_img_2D('test0.png')

    threshold(src_img, 100)

    # Sobel operator
    filter = np.array([(-1, -2, -1),
                       ( 0,  0,  0),
                       ( 1,  2,  1)
    ])
    conv_img = convole(src_img, filter)

    # Output
    save_img_from_array(conv_img, out_name = 'conv_out_img.png')


# Executing test
execute(question3_test)
