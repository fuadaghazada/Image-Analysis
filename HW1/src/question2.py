from pprint import pprint
import numpy as np

from mainlib.file import load_img_2D, save_img_from_array
from mainlib.morph_operations import dilation, erosion
from mainlib.threshold import adaptive_threshold

from other.loading import execute

######## QUESTION 2 ########

def question2_test():

    src_img = load_img_2D('sonnet.png')

    thr_img = adaptive_threshold(src_img, (5, 5), 5, 0)

    strc_el = np.ones((1, 3)).astype(np.uint8) * 255
    out_img = erosion(thr_img, strc_el)

    strc_el = np.ones((1, 1)).astype(np.uint8) * 255
    out_img = dilation(thr_img, strc_el)

    # Output
    save_img_from_array(out_img, out_name = 'sonnet_out.png')


# Executing test
execute(question2_test)
