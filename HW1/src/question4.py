from pprint import pprint
import numpy as np

from mainlib.file import load_img_2D, save_img_from_array
from mainlib.ffiltering import gaussian_filter

from other.loading import execute

######## QUESTION 4 ########

def question4_test():

    src_img = load_img_2D('test1.png')

    # Applying Gaussian filter with the given sigma
    out_img = gaussian_filter(src_img, 9)

    # Output
    save_img_from_array(out_img.astype(np.uint8), out_name = 'glp_out.png')


# Executing test
execute(question4_test)
