from pprint import pprint
import numpy as np

from mainlib.file import load_img_2D, save_img_from_array
from mainlib.morph_operations import dilation, erosion
from mainlib.threshold import adaptive_threshold

from other.loading import execute

######## QUESTION 2 ########

def question2_test():

    src_img = load_img_2D('sonnet.png')

    thr_img = adaptive_threshold(src_img, (7, 7), 7, 0)

    strc_el = np.ones((3, 1)).astype(np.uint8) * 255

    out_img = erosion(thr_img, strc_el)

    # out_img = dilation(out_img, strc_el2)

    # Output
    save_img_from_array(out_img, out_name = 'sonnet_out.png')


# Executing test
execute(question2_test)
