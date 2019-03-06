from pprint import pprint
import numpy as np

from mylib.file import load_img_2D, save_img_from_array
from mylib.morph_operations import dilation, erosion
from mylib.threshold import adaptive_threshold

######## QUESTION 2 ########
src_img = load_img_2D('sonnet.png')

thr_img = adaptive_threshold(src_img, (100, 75), 1)

strc_el = np.ones((3, 1)).astype(np.uint8) * 255
#
out_img = erosion(thr_img, strc_el)
out_img = dilation(out_img, strc_el)

# Output
save_img_from_array(out_img, out_name = 'sonnet_out.png')
