from pprint import pprint
import numpy as np

from mainlib.file import load_img_2D, save_img_from_array
from mainlib.threshold import threshold
from mainlib.sfiltering import convole

from other.loading import execute

######## QUESTION 3 ########

def question3_test():

    src_img = load_img_2D('test5.png')

    # Sobel operator
    filter1 = np.array([(-1, -2, -1),
                       ( 0,  0,  0),
                       ( 1,  2,  1)
    ])

    filter2 = np.array([(-1, 0, 1),
                       (-2, 0, 2),
                       (-1, 0, 1)
    ])

    out_img1 = convole(src_img, filter1)
    out_img2 = convole(src_img, filter2)

    # Output
    out_img1.save('../out/conv_out_img1.png')
    out_img2.save('../out/conv_out_img2.png')


# Executing test
execute(question3_test)
