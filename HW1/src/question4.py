from pprint import pprint
import numpy as np

from mainlib.file import load_img_2D, save_img_from_array
from mainlib.ffiltering import gaussian_filter

from other.loading import execute

######## QUESTION 4 ########

def question4_test():

    src_img = load_img_2D('test1.png')

    # Applying Gaussian filter with the given sigma
    fft_src_img, g_func, fft_filter_func, out, fft_out = gaussian_filter(src_img, 9)

    # Output
    save_img_from_array(src_img.astype(np.uint8), out_name = 'q4/gaussian_input.png')
    save_img_from_array(fft_src_img.astype(np.uint8), out_name = 'q4/fft_input.png')
    save_img_from_array(g_func.astype(np.uint8), out_name = 'q4/gaussian_func.png')
    save_img_from_array(fft_filter_func.astype(np.uint8), out_name = 'q4/fft_gaussian_func.png')
    save_img_from_array(out.astype(np.uint8), out_name = 'q4/gaussian_output.png')
    save_img_from_array(fft_out.astype(np.uint8), out_name = 'q4/fft_gaussian_output.png')


# Executing test
execute(question4_test)
